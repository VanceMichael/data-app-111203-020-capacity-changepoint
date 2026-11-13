"""按设备（asset）的分段趋势分析。

每台设备单独拟合，不跨设备拼接观测。模型为

    throughput = b0 + b1 * (day - day_ref)
                 + sum_j delta_j * hinge_j(day)
                 + gamma * (temperature - temperature_ref)
                 + sum_s jump_s * 1[day >= s] + error

- ``knot`` 是斜率变化点（连续折线），只允许出现在记录到的运行日；
  ``hinge_j`` 用折叠参数化表示相邻两段斜率之差。
- ``service_day`` 是已知维护日，只作为水平跳变，不进入拐点候选。
- ``gamma`` 在每个训练窗口内用该窗口的数据估计；最终拟合可使用同 model 的
  经验贝叶斯先验（只共享先验强度，不共享观测）。只有一台设备时不加先验。
- ``hours <= 0`` 的停机日不进入似然，也不把 throughput 当作 0。
- 单次离群先被标出并从段数选择中剔除；其余观测按有效运行小时加权，
  稳健拟合用固定尺度的 Huber 重加权。

复杂度惩罚形如 ``n*log(HuberLoss/n) + corr*(p + penalty*K)*log(n)``，
每个拐点除 1 个参数外再付出变点位置的成本，``corr`` 是小样本修正。
"""

import itertools
import math
import random
import statistics
from collections import Counter, defaultdict

from .linalg import independent_columns, inverse, solve

PRODUCT = 'capacity-changepoint'
SCHEMA_VERSION = 2
UNIT = '件每小时'

DEFAULT_OPTIONS = {
    'min_side_hours': 24.0,
    'min_side_days': 3,
    # 候选拐点两侧紧邻观测的最大允许空档（天）。
    'min_side_gap_days': 4,
    'min_segment_hours': 24.0,
    'min_segment_days': 3,
    'min_observations': 7,
    'max_knots': 2,
    'change_point_penalty': 3.0,
    'max_small_sample_correction': 1.5,
    # 维护日之后的正向窗口内，斜率转折与维护恢复无法区分，不作拐点候选。
    'service_window_days': 1,
    'degradation_threshold': 0.25,
    'confirmation_min_drop': 2.0,
    'min_confirmation_observations': 3,
    'huber_tuning': 1.345,
    'irls_iterations': 6,
    # 稳健拟合的尺度下限（件每小时）。合成数据噪声约 0.5；下限防止
    # 曲线完全拟合后尺度塌缩、把正常波动当成离群。
    'robust_floor': 0.5,
    # 空模型残差超过阈值倍数（稳健尺度）的观测按单次离群处理。
    'outlier_threshold': 3.5,
    'bootstrap_replicates': 80,
    'bootstrap_seed': 20240601,
    'stability_min_share': 0.6,
    'stability_max_width': 2,
    # 段数选择中，最优零拐点结构与最优带拐点结构的分差小于该值时，
    # 不宣布“无趋势转折”，而是标记为证据不足。
    'ambiguity_margin': 2.0,
    'temperature_prior_floor': 0.05,
    'rolling_min_history': 7,
}


def resolve_options(overrides=None):
    options = dict(DEFAULT_OPTIONS)
    for key, value in (overrides or {}).items():
        if key not in options:
            raise ValueError('未知分析选项：' + key)
        if key in ('min_side_days', 'min_segment_days', 'min_observations',
                   'max_knots', 'min_confirmation_observations',
                   'bootstrap_replicates', 'rolling_min_history',
                   'service_window_days', 'min_side_gap_days'):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError('选项需要非负整数：' + key)
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('选项需要有限数值：' + key)
        options[key] = value
    if options['max_knots'] < 0 or options['max_knots'] > 3:
        raise ValueError('max_knots 必须在 0 到 3 之间')
    if options['stability_min_share'] <= 0 or options['stability_min_share'] > 1:
        raise ValueError('stability_min_share 必须在 (0, 1] 之间')
    if options['min_side_hours'] < 0 or options['min_segment_hours'] < 0:
        raise ValueError('最小运行小时不能为负')
    return options


def usable(rows):
    """运行观测：只保留 hours>0 的日期。停机日不是零产能。"""
    return [row for row in rows if row['hours'] > 0.0]


def observation_days(rows):
    return sorted({row['day'] for row in rows})


def service_events(rows, min_side_days=0):
    """已知维护日：service_day != -1，出现在窗口内，且两侧都有运行观测。

    ``min_side_days`` 要求维护日两侧各有足够运行日；窗口太短时维护跳变
    与趋势无法区分，此时不把它放进模型，只在候选淘汰原因里说明。
    """
    days = observation_days(rows)
    if not days:
        return []
    first, last = days[0], days[-1]
    events = sorted({row['service_day'] for row in rows if row['service_day'] != -1})
    result = []
    for event in events:
        if not first <= event <= last + 1:
            continue
        before = {row['day'] for row in rows
                  if row['day'] < event and row['hours'] > 0.0}
        after = {row['day'] for row in rows
                 if row['day'] >= event and row['hours'] > 0.0}
        if len(before) >= max(1, min_side_days) and len(after) >= max(1, min_side_days):
            result.append(event)
    return result


def dropped_service_events(rows, min_side_days):
    """窗口太短、无法单独估计的维护事件，用于显式说明而不是静默忽略。"""
    days = observation_days(rows)
    if not days:
        return []
    first, last = days[0], days[-1]
    kept = set(service_events(rows, min_side_days))
    output = []
    for event in sorted({row['service_day'] for row in rows if row['service_day'] != -1}):
        if not first <= event <= last + 1 or event in kept:
            continue
        output.append(event)
    return output


def _design(rows, knots, services):
    """构造设计矩阵；返回 None 表示该结构在数据上不可识别。"""
    rows = list(rows)
    days = [row['day'] for row in rows]
    first = min(days)
    temperature_ref = statistics.fmean(row['temperature'] for row in rows)
    intercept = [1.0] * len(rows)
    trend = [float(row['day'] - first) for row in rows]
    temperature = [row['temperature'] - temperature_ref for row in rows]
    jumps = [[1.0 if row['day'] >= event else 0.0 for row in rows] for event in services]
    # 折线基（hinge）在数值上几乎共线；改用“折叠”参数化：
    # max(0, day-knot) - max(0, day-knot_prev)，最后一节为整条折线。
    # 它与 hinge 张成同一空间，但条件数好得多，斜率方差也更容易估计。
    unfolded = [[max(0.0, float(row['day'] - knot)) for row in rows] for knot in knots]
    hinges = []
    for position, column in enumerate(unfolded):
        if position + 1 < len(unfolded):
            hinges.append([column[i] - unfolded[position + 1][i] for i in range(len(rows))])
        else:
            hinges.append(list(column))

    # 先放入已知效应，后放待估拐点；被判定为共线的拐点直接使候选失效。
    kept_services = [index for index in independent_columns([intercept, trend, temperature] + jumps)
                     if index >= 3]
    base = [intercept, trend, temperature] + [jumps[index - 3] for index in kept_services]
    kept_hinges = [index for index in independent_columns(base + hinges) if index >= len(base)]
    if len(kept_hinges) != len(hinges):
        return None
    columns = base + hinges
    return {
        'columns': columns,
        'first_day': first,
        'temperature_ref': temperature_ref,
        'hinge_columns': list(range(len(base), len(base) + len(hinges))),
        'temperature_column': 2,
        'jump_columns': list(range(3, 3 + len(kept_services))),
        'jump_days': [services[index - 3] for index in kept_services],
        'dropped_services': [event for position, event in enumerate(services)
                             if position + 3 not in kept_services],
    }


def noise_scale(rows, options):
    """用二阶差分的中位绝对值估计噪声尺度。

    二阶差分消去线性趋势，中位数对单个离群点不敏感；所有候选结构共用
    这一个尺度，稳健损失才可跨模型比较。
    """
    ordered = sorted(rows, key=lambda row: (row['day'], row['observation_id']))
    if len(ordered) >= 3:
        seconds = [ordered[i + 2]['throughput'] - 2.0 * ordered[i + 1]['throughput']
                   + ordered[i]['throughput'] for i in range(len(ordered) - 2)]
        spread = statistics.median(abs(value) for value in seconds)
        estimate = spread / (0.6744897501960817 * math.sqrt(6.0))
    elif len(ordered) >= 2:
        differences = [ordered[i + 1]['throughput'] - ordered[i]['throughput']
                       for i in range(len(ordered) - 1)]
        spread = statistics.median(abs(value) for value in differences)
        estimate = spread / (0.6744897501960817 * math.sqrt(2.0))
    else:
        estimate = 0.0
    return max(estimate, options['robust_floor'])


def _huber_weights(base_weights, residuals, centre, scale, tuning):
    output = []
    for weight, residual in zip(base_weights, residuals):
        ratio = abs(residual - centre) / scale
        factor = 1.0 if ratio <= tuning else tuning / ratio
        output.append(weight * factor)
    return output


def _huber_loss(base_weights, residuals, centre, scale, tuning):
    total = 0.0
    for weight, residual in zip(base_weights, residuals):
        size = abs(residual - centre) / scale
        total += weight * (0.5 * size * size if size <= tuning
                           else tuning * (size - 0.5 * tuning))
    return total


def _weighted_fit(columns, y, weights, prior_column, prior_mean, prior_precision):
    size = len(columns)
    normal = [[0.0] * size for _ in range(size)]
    rhs = [0.0] * size
    for row_index, weight in enumerate(weights):
        if weight == 0.0:
            continue
        target = y[row_index]
        values = [column[row_index] for column in columns]
        for a in range(size):
            va = values[a]
            if va == 0.0:
                continue
            weighted = weight * va
            rhs[a] += weighted * target
            row_a = normal[a]
            for b in range(a, size):
                row_a[b] += weighted * values[b]
    for a in range(size):
        for b in range(a):
            normal[a][b] = normal[b][a]
    if prior_precision > 0.0:
        normal[prior_column][prior_column] += prior_precision
        rhs[prior_column] += prior_precision * prior_mean
    trace = sum(normal[i][i] for i in range(size))
    if trace <= 0.0 or not math.isfinite(trace):
        return None
    jitter = max(trace * 1e-12, 1e-12)
    for i in range(size):
        normal[i][i] += jitter
    beta = solve(normal, rhs)
    if beta is None:
        return None
    return beta, normal


def fit_structure(rows, knots, services, options, prior=None, design=None,
                  scale=None):
    """拟合一个已知分段结构，返回系数、稳健权重、协方差与可比损失。

    ``design`` 可由调用方缓存：重采样只改变 throughput，设计矩阵不变。
    ``scale`` 也由调用方固定，使不同候选结构的损失处在同一尺度上。
    """
    if not rows:
        return None
    if design is None:
        design = _design(rows, knots, services)
    if design is None:
        return None
    columns = design['columns']
    if len(columns[0]) != len(rows):
        raise ValueError('设计矩阵与观测行数不一致')
    y = [row['throughput'] for row in rows]
    mean_hours = statistics.fmean(row['hours'] for row in rows)
    base_weights = [row['hours'] / mean_hours for row in rows]
    fixed_scale = noise_scale(rows, options) if scale is None else scale
    prior_mean, prior_precision = (prior if prior else (0.0, 0.0))
    prior_column = design['temperature_column']
    tuning = options['huber_tuning']
    weights = list(base_weights)
    for iteration in range(max(1, options['irls_iterations'])):
        fitted = _weighted_fit(columns, y, weights, prior_column, prior_mean,
                               prior_precision)
        if fitted is None:
            return None
        beta, normal = fitted
        residuals = [y[i] - sum(columns[c][i] * beta[c] for c in range(len(beta)))
                     for i in range(len(y))]
        centre = statistics.median(residuals)
        updated = _huber_weights(base_weights, residuals, centre, fixed_scale, tuning)
        if iteration and max(abs(a - b) for a, b in zip(updated, weights)) <= 1e-4:
            weights = updated
            break
        weights = updated
    fitted = _weighted_fit(columns, y, weights, prior_column, prior_mean,
                           prior_precision)
    if fitted is None:
        return None
    beta, normal = fitted
    residuals = [y[i] - sum(columns[c][i] * beta[c] for c in range(len(beta)))
                 for i in range(len(y))]
    centre = statistics.median(residuals)
    wrss = math.fsum(w * r * r for w, r in zip(weights, residuals))
    loss = _huber_loss(base_weights, residuals, centre, fixed_scale, tuning)
    effective = math.fsum(weights)
    parameters = len(beta)
    covariance = None
    if effective > parameters:
        covariance = inverse(normal)
        if covariance is not None:
            sigma2 = wrss / (effective - parameters)
            covariance = [[sigma2 * covariance[a][b] for b in range(parameters)]
                          for a in range(parameters)]
    return {
        'design': design,
        'beta': beta,
        'covariance': covariance,
        'weights': weights,
        'residuals': residuals,
        'wrss': wrss,
        'loss': loss,
        'loss_scale': fixed_scale,
        'effective_n': effective,
        'parameters': parameters,
        'knots': list(knots),
        'services': design['jump_days'],
        'robust_scale': fixed_scale,
        'y': y,
        'rows': list(rows),
    }


def flag_outliers(rows, options, scale=None):
    """标出单次离群观测：某个点相对相邻点的局部跳变远超稳健尺度。

    用局部比较而不是对整条曲线拟合的残差，是因为曲线本身可能有真实
    转折；局部比较只对孤立尖峰敏感。被标出的观测不进入结构选择，
    从而单次离群不能制造拐点。
    """
    if len(rows) < options['min_observations']:
        return []
    ordered = sorted(rows, key=lambda row: (row['day'], row['observation_id']))
    services = service_events(ordered, options['min_side_days'])
    fit = fit_structure(ordered, [], services, options)
    if fit is None:
        return []
    residuals = fit['residuals']
    if scale is None:
        centre = statistics.median(residuals)
        spread = statistics.median(abs(value - centre) for value in residuals)
        scale = max(1.4826 * spread, options['robust_floor'])
    # 尖峰必须同时明显偏离左右邻居，而且偏离方向一致。转折或台阶处
    # 左右两侧本身差值就大，用同一阈值会把真实结构误判成离群。
    # 这里再要求“偏离量明显大于左右两侧之间的水平差”，把台阶排除掉。
    flagged = []
    for index, residual in enumerate(residuals):
        left = [residuals[p] for p in (index - 2, index - 1) if 0 <= p < len(residuals)]
        right = [residuals[p] for p in (index + 1, index + 2) if 0 <= p < len(residuals)]
        if not left or not right:
            continue
        left_level = statistics.median(left)
        right_level = statistics.median(right)
        local = residual - left_level
        other = residual - right_level
        level_gap = abs(left_level - right_level)
        spike = min(abs(local), abs(other))
        # 台阶：相邻点之间整体平移，单点两侧的残差方向相同、幅度相当。
        # 真正的尖峰要求该点相对“两侧各自水平”都突出，而两侧水平相近。
        if (spike > options['outlier_threshold'] * scale
                and (local > 0) == (other > 0)
                and spike > 1.5 * level_gap):
            flagged.append(index)
    return flagged


def eligible_knots(rows, services, options):
    """列出满足两侧/中间段最小有效运行小时与运行日的候选拐点日。"""
    days = observation_days(rows)
    if len(days) < 2:
        return [], []
    events = set(services)
    eligible, rejected = [], []
    for position, day in enumerate(days[1:], start=1):
        left = [row for row in rows if row['day'] < day]
        right = [row for row in rows if row['day'] >= day]
        left_hours = math.fsum(row['hours'] for row in left)
        right_hours = math.fsum(row['hours'] for row in right)
        left_days = len({row['day'] for row in left})
        right_days = len({row['day'] for row in right})
        reasons = []
        if left_hours < options['min_side_hours']:
            reasons.append('左侧有效运行小时不足')
        if right_hours < options['min_side_hours']:
            reasons.append('右侧有效运行小时不足')
        if left_days < options['min_side_days']:
            reasons.append('左侧有效运行日不足')
        if right_days < options['min_side_days']:
            reasons.append('右侧有效运行日不足')
        # 变点要落在观测密集处：两侧最近一次观测的空档都不能过大，
        # 否则折点等于拿两段相隔很远的数据拼出来。
        left_gap = day - days[position - 1]
        right_gap = days[position] - day if position < len(days) - 1 else 1
        right_gap = days[position + 1] - days[position] if position + 1 < len(days) else right_gap
        if left_gap > options['min_side_gap_days']:
            reasons.append('左侧紧邻空档过大')
        if right_gap > options['min_side_gap_days']:
            reasons.append('右侧紧邻空档过大')
        if day in events:
            reasons.append('与已知维护日重合，按水平跳变处理，不作为趋势拐点')
        elif any(0 < day - event <= options['service_window_days'] for event in events):
            reasons.append('紧邻维护日，维护恢复与趋势转折无法区分')
        record = {
            'day': day,
            'left_hours': left_hours,
            'right_hours': right_hours,
            'left_days': left_days,
            'right_days': right_days,
            'eligible': not reasons,
            'reasons': reasons,
        }
        (eligible if not reasons else rejected).append(record)
    return eligible, rejected


def _segment_ok(rows, low, high, options, key):
    """区间 [low, high) 的最小样本要求。"""
    members = [row for row in rows if low <= row['day'] < high]
    hours = math.fsum(row['hours'] for row in members)
    days = len({row['day'] for row in members})
    if hours < options[key + '_hours'] or days < options[key + '_days']:
        return False
    return True


def _score(fit, options, knots):
    """稳健信息准则：固定尺度下的 Huber 损失 + 显式复杂度惩罚。

    小样本下 log(n) 惩罚偏弱（n 只有十余个观测时凹得很厉害），
    因此对参数个数再加一个 n/(n-p) 的小样本修正，避免短窗口过拟合。
    """
    n = fit['effective_n']
    parameters = fit['parameters']
    if n <= parameters + 1:
        return None
    penalty = parameters + options['change_point_penalty'] * len(knots)
    # 小样本修正有上限：短窗口里 n-p 很小，不设上限会把惩罚放大到无法
    # 选择任何结构，反而让滚动验证永远提前报不出提示。
    correction = min(n / (n - parameters), options['max_small_sample_correction'])
    return n * math.log(max(fit['loss'] / n, 1e-12)) + correction * penalty * math.log(n)


def select_structure(rows, services, options, prior=None, max_knots=None, cache=None):
    """按显式复杂度惩罚选择拐点个数与位置。

    先在空模型上标出单次离群并剔除；离群点不参与段数选择，也不作为
    拐点候选。
    """
    limit = options['max_knots'] if max_knots is None else min(max_knots, options['max_knots'])
    flagged = flag_outliers(rows, options)
    excluded_ids = {id(rows[index]) for index in flagged}
    cleaned = [row for row in rows if id(row) not in excluded_ids]
    outliers = [rows[index] for index in flagged]
    if len(cleaned) >= options['min_observations']:
        rows = cleaned
    else:
        outliers = []
    eligible, rejected = eligible_knots(rows, services, options)
    candidates = [record['day'] for record in eligible]
    scale = noise_scale(rows, options)
    evaluations = []
    for count in range(0, min(limit, len(candidates)) + 1):
        best = None
        for combo in itertools.combinations(candidates, count):
            if count >= 2:
                bounds = [min(row['day'] for row in rows)] + list(combo) + [max(row['day'] for row in rows) + 1]
                if not all(_segment_ok(rows, bounds[i], bounds[i + 1], options, 'min_segment')
                           for i in range(len(bounds) - 1)):
                    continue
            design = None
            if cache is not None:
                key = (tuple(combo), tuple(services))
                if key not in cache:
                    cache[key] = _design(rows, list(combo), services)
                design = cache[key]
                if design is None:
                    continue
            fit = fit_structure(rows, list(combo), services, options, prior,
                                design=design, scale=scale)
            if fit is None:
                continue
            score = _score(fit, options, combo)
            if score is None:
                continue
            record = {'knots': list(combo), 'fit': fit, 'score': score}
            if best is None or score < best['score']:
                best = record
        if best is not None:
            evaluations.append(best)
    if not evaluations:
        return None
    chosen = min(evaluations, key=lambda item: (item['score'], len(item['knots'])))
    return {
        'chosen': chosen,
        'evaluations': evaluations,
        'eligible': candidates,
        'rejected': rejected,
        'services': services,
        'outliers': outliers,
    }


Z_95 = 1.959963984540054


def _slope_details(fit, index):
    """第 index 段斜率及其系数区间（不是日期区间）。

    折叠参数化下，第 index 段斜率为 b1 + sum(前 index 个折叠系数)，
    只是把 hinge 参数里“相邻差值”显式写成独立系数。
    """
    beta = fit['beta']
    covariance = fit['covariance']
    hinges = fit['design']['hinge_columns']
    terms = [1] + hinges[:index]
    slope = beta[1] + math.fsum(beta[column] for column in hinges[:index])
    variance = None
    if covariance is not None:
        variance = math.fsum(covariance[a][b] for a in terms for b in terms)
        variance = max(variance, 0.0)
    half = Z_95 * math.sqrt(variance) if variance is not None else None
    return slope, (None if half is None else [slope - half, slope + half])


def _segment_rows(rows, low, high):
    return [row for row in rows if low <= row['day'] < high]


def segment_table(fit, rows, options):
    """输出每段起止、有效样本与斜率（含斜率系数的 95% 区间）。"""
    knots = fit['knots']
    days = observation_days(rows)
    first, last = days[0], days[-1] + 1
    bounds = [first] + knots + [last]
    table = []
    for index in range(len(bounds) - 1):
        low, high = bounds[index], bounds[index + 1]
        members = _segment_rows(rows, low, high)
        slope, interval = _slope_details(fit, index)
        table.append({
            'segment': index,
            'start_day': low,
            'end_day': high - 1,
            'observations': len(members),
            'runtime_hours': math.fsum(row['hours'] for row in members),
            'slope': slope,
            'slope_interval': interval,
            'interval_note': '斜率系数的置信区间，不是拐点日期区间',
        })
    return table


def _resample_rows(fit, rng):
    """残差重抽样：保留拟合结构，用拟合值 ± 残差生成伪数据。

    残差本身来自稳健拟合，量级已经反映该设备的噪声；用 Rademacher
    符号只打乱方向，不重新估计尺度。
    """
    beta = fit['beta']
    generated = []
    for position, row in enumerate(fit['rows']):
        values = [column[position] for column in fit['design']['columns']]
        expected = math.fsum(b * v for b, v in zip(beta, values))
        shock = fit['residuals'][position] * (1.0 if rng.random() < 0.5 else -1.0)
        generated.append(dict(row, throughput=expected + shock))
    return generated


def bootstrap_stability(rows, services, selection, options, prior=None, seed=None):
    """对选择后的拐点位置做残差重抽样，报告稳定日期范围与证据不足状态。"""
    replicas = options['bootstrap_replicates']
    chosen = selection['chosen']
    knots = chosen['knots']
    fit = chosen['fit']
    result = {
        'requested': replicas,
        'replicates': 0,
        'method': '残差重抽样（稳健尺度），每次重抽样重新选择段数',
        'days': [],
        'superseded': 0,
        'share': 0.0,
        'knot_count_days': [],
        'summary': None,
        'status': '证据不足',
        'note': '拐点日期范围来自重采样分布，不能把回归斜率区间当作日期区间。',
    }
    if not knots:
        result['status'] = '未检出趋势转折'
        result['note'] = '所选结构无拐点，不需要重采样日期稳定性。'
        return result
    if replicas <= 0:
        result['status'] = '未评估'
        result['note'] = '本次运行关闭了重采样；日期稳定性未评估，不能读作“证据不足”。'
        return result
    rng = random.Random(options['bootstrap_seed'] if seed is None else seed)
    day_hits = [Counter() for _ in knots]
    count_hits = Counter()
    cache = {}
    for _ in range(replicas):
        generated = _resample_rows(fit, rng)
        try:
            resampled = select_structure(generated, services, options, prior,
                                         cache=cache)
        except (ValueError, ZeroDivisionError):
            resampled = None
        result['replicates'] += 1
        if resampled is None:
            count_hits[None] += 1
            continue
        count_hits[len(resampled['chosen']['knots'])] += 1
        positions = resampled['chosen']['knots']
        if len(positions) != len(knots):
            result['superseded'] += 1
            continue
        for index, value in enumerate(positions):
            day_hits[index][value] += 1
    total = max(result['replicates'], 1)
    result['share'] = sum(
        counters.get(knots[position], 0) for position, counters in enumerate(day_hits)
    ) / max(len(knots) * total, 1)
    counts = result['knot_count_days'] = sorted(
        ({'knots': None if count is None else count, 'replicates': hits}
         for count, hits in count_hits.items()),
        key=lambda item: (-1 if item['knots'] is None else item['knots']))
    dominant = None
    if counts:
        dominant = max(counts, key=lambda item: (item['knots'] is not None,
                                                 item['replicates']))
    days = []
    for index, hits in enumerate(day_hits):
        ordered = sorted(hits.items())
        # 统一用全部重抽样次数作分母，避免“段数变了”的样本既进分子又改分母。
        stable = sorted(day for day, count in ordered
                        if count >= options['stability_min_share'] * total)
        if not ordered:
            days.append({'knot': index, 'day': knots[index], 'mode': None,
                         'range': None, 'share': 0.0, 'status': '证据不足'})
            continue
        mode, _ = max(ordered, key=lambda item: item[1])
        low, high = (min(stable), max(stable)) if stable else (mode, mode)
        width = high - low
        day_share = sum(count for day, count in ordered if day == knots[index]) / total
        if day_share >= options['stability_min_share'] \
                and width <= options['stability_max_width'] and stable:
            status = '稳定'
        else:
            status = '证据不足'
        days.append({
            'knot': index,
            'day': knots[index],
            'mode': mode,
            'range': [low, high],
            'width_days': width,
            'share': day_share,
            'support': [{'day': day, 'count': count} for day, count in ordered],
            'status': status,
        })
    result['days'] = days
    # 先判日期本身是否集中，再判结构是否稳定；不把“段数变了”自动算作日期不稳定。
    if all(item['status'] == '稳定' for item in days):
        result['status'] = '稳定'
    elif all(item['share'] >= options['stability_min_share'] for item in days):
        result['status'] = '基本稳定'
    else:
        result['status'] = '证据不足'
    result['structure_share'] = (dominant['replicates'] / total) if dominant else 0.0
    result['structure_stable'] = (
        dominant is not None and dominant['knots'] == len(knots)
        and result['structure_share'] >= options['stability_min_share'])
    if not result['structure_stable'] and result['status'] == '稳定':
        result['status'] = '基本稳定'
    # 主失败模式写清楚：是日期散布、还是段数本身不稳。
    if result['status'] != '稳定':
        if not result['structure_stable']:
            result['reason'] = '重采样中多数未选择同样的段数结构'
        else:
            result['reason'] = '重采样中拐点日期分散，日期范围不够窄'
    else:
        result['reason'] = None
    return result


def _temperature_estimate(rows, services, options, prior=None):
    """第一阶段：逐设备估计温度系数，供同 model 的经验贝叶斯先验使用。"""
    if len(rows) < options['min_observations']:
        return None
    selection = select_structure(rows, services, options, prior)
    if selection is None:
        return None
    fit = selection['chosen']['fit']
    column = fit['design']['temperature_column']
    estimate = fit['beta'][column]
    variance = fit['covariance'][column][column] if fit['covariance'] is not None else None
    return {'estimate': estimate, 'variance': variance, 'fit': fit, 'selection': selection}


def model_priors(asset_rows, options, model_of):
    """由第一阶段估计形成同 model 的先验均值与精度，不合并任何原始观测。"""
    first_pass = {}
    for asset, rows in asset_rows.items():
        services = service_events(rows)
        estimate = _temperature_estimate(rows, services, options)
        if estimate is not None and estimate['variance'] is not None:
            first_pass[asset] = estimate
    priors = {}
    groups = defaultdict(list)
    for asset, estimate in first_pass.items():
        groups[model_of[asset]].append(estimate)
    for model in sorted(groups):
        members = groups[model]
        means = [item['estimate'] for item in members]
        variances = [item['variance'] for item in members]
        if len(members) >= 2:
            mean = statistics.fmean(means)
            between = statistics.variance(means)
            within = statistics.fmean(variances)
            tau2 = max(between - within, options['temperature_prior_floor'])
            priors[model] = {
                'mean': mean,
                'precision': 1.0 / tau2,
                'tau2': tau2,
                'assets': len(members),
                'applied': True,
                'method': '经验贝叶斯（组内均值 + 去噪组间方差）',
            }
        else:
            # 只有一台设备时，组均值就是它自己的估计，再做收缩等于用自己
            # 证明自己；这时不施加先验，只记录为什么没有共享。
            priors[model] = {
                'mean': means[0],
                'precision': 0.0,
                'tau2': None,
                'assets': 1,
                'applied': False,
                'method': '单一设备，不施加组先验',
            }
    return priors, first_pass


def _onset(fit, options):
    """从分段斜率推断持续退化起点；首段已明显为负时报告“起点早于窗口”。"""
    table = segment_table(fit, fit['rows'], options)
    slopes = [item['slope'] for item in table]
    threshold = options['degradation_threshold']
    # 只有首段明显低于阈值才说“起点早于窗口”；-0.27 这种贴线值不能算。
    if slopes[0] < -2.0 * threshold:
        return {'type': 'before_window', 'day': table[0]['start_day'],
                'note': '首个可见段已明显低于退化阈值，转折可能早于本窗口'}
    for index, slope in enumerate(slopes):
        if slope > -threshold:
            continue
        if index > 0 and all(value <= -threshold for value in slopes[index:]):
            return {'type': 'knot', 'day': table[index]['start_day'],
                    'segment': index, 'note': None}
    return None


def _confirmation(rows, fit, onset, options):
    """用退化起点之后的运行观测检查累计降幅是否足以确认。"""
    if onset is None:
        return {'confirmed': False, 'reason': '没有满足阈值的持续退化段',
                'observations': 0}
    day = onset['day']
    after = sorted((row for row in rows if row['day'] >= day),
                   key=lambda row: (row['day'], row['observation_id']))
    if len(after) < options['min_confirmation_observations']:
        return {'confirmed': False, 'reason': '退化起点之后的有效观测不足',
                'observations': len(after)}
    slope = None
    for item in segment_table(fit, fit['rows'], options):
        if item['start_day'] <= day <= item['end_day']:
            slope = item['slope']
            break
    if slope is None:
        return {'confirmed': False, 'reason': '无法定位退化段',
                'observations': len(after)}
    first, last = after[0], after[-1]
    span = last['day'] - first['day']
    observed_drop = first['throughput'] - last['throughput']
    expected_drop = -slope * span
    confirm = (expected_drop >= options['confirmation_min_drop']
               and observed_drop >= 0.5 * options['confirmation_min_drop'])
    return {
        'confirmed': bool(confirm),
        'slope': slope,
        'from_day': first['day'],
        'to_day': last['day'],
        'span_days': span,
        'expected_drop': expected_drop,
        'observed_drop': observed_drop,
        'observations': len(after),
        'reason': None if confirm else '退化幅度尚未达到确认下限',
    }


def _ambiguous(selection, options):
    """零拐点胜出但优势很小：结构本身不确定。"""
    chosen = selection['chosen']
    if chosen['knots']:
        return False
    alternatives = [item['score'] for item in selection['evaluations'] if item['knots']]
    if not alternatives:
        return False
    return min(alternatives) - chosen['score'] < options['ambiguity_margin']


def _evidence(rows, selection, stability, options, onset, confirmation):
    """综合给出证据状态，避免仅凭一个显著性水平宣布退化。"""
    if selection is None:
        return '样本不足'
    chosen = selection['chosen']
    fit = chosen['fit']
    if fit['effective_n'] <= fit['parameters']:
        return '样本不足'
    knots = chosen['knots']
    if not knots:
        if _ambiguous(selection, options):
            return '证据不足'
        return '无趋势转折'
    if stability['status'] == '未评估':
        # 跳过重采样时只能给出结构判断，日期稳定性未知。
        return '退化（日期稳定性未评估）' if onset and confirmation['confirmed'] \
            else '证据不足'
    if stability['status'] == '证据不足':
        return '证据不足'
    if onset is None:
        return '无持续退化'
    if not confirmation['confirmed']:
        return '退化未确认'
    if stability['status'] == '基本稳定':
        return '疑似退化（日期待定）'
    return '持续退化'


def diagnose_rows(rows, model, options, prior=None, excluded_rows=None):
    """对单台设备的运行观测做完整诊断。

    ``rows`` 是 hours>0 的运行观测；``excluded_rows`` 是同一窗口的停机日，
    只用于统计被排除的天数和服务事件位置，不进入似然。
    """
    excluded_rows = list(excluded_rows or [])
    days = observation_days(rows)
    result = {
        'asset': rows[0]['asset'] if rows else None,
        'model': model,
        'window': None,
        'usable_days': 0,
        'usable_observations': len(rows),
        'excluded_days': 0,
        'services': [],
        'dropped_services': [],
        'outliers': [],
        'temperature': None,
        'candidates': [],
        'model_selection': {'chosen_knots': [], 'candidate_days': [],
                            'scores': [], 'services': [], 'status': '样本不足'},
        'segments': [],
        'service_effects': [],
        'onset': None,
        'confirmation': None,
        'stability': None,
        'evidence': '样本不足',
        'notes': [],
    }
    if rows:
        result['window'] = [min(row['day'] for row in rows), max(row['day'] for row in rows)]
    result['usable_days'] = len(days)
    result['excluded_days'] = len(observation_days(excluded_rows))
    services = service_events(rows + excluded_rows, options['min_side_days'])
    dropped = dropped_service_events(rows + excluded_rows, options['min_side_days'])
    result['services'] = services
    result['dropped_services'] = dropped
    if dropped:
        result['notes'].append('维护日 ' + ','.join(str(day) for day in dropped)
                               + ' 两侧运行日不足，窗口内不单独估计跳变')
    if len(rows) < options['min_observations']:
        result['notes'].append('有效运行观测少于最少要求，无法估计分段趋势')
        return result
    selection = select_structure(rows, services, options, prior)
    if selection is None:
        result['notes'].append('候选结构在数值上不可识别')
        return result
    chosen = selection['chosen']
    fit = chosen['fit']
    result['outliers'] = [{'day': row['day'], 'observation_id': row['observation_id'],
                           'throughput': row['throughput']}
                          for row in selection.get('outliers', [])]
    if result['outliers']:
        result['notes'].append('已排除单次离群观测：' + ','.join(
            str(item['day']) for item in result['outliers'])
            + '；这些观测不参与段数选择')
    result['candidates'] = sorted(
        [{'day': item['day'], 'eligible': False, 'reasons': item['reasons']}
         for item in selection['rejected']]
        + [{'day': day, 'eligible': True, 'reasons': []} for day in selection['eligible']],
        key=lambda item: item['day'])
    result['model_selection'] = {
        'chosen_knots': chosen['knots'],
        'candidate_days': selection['eligible'],
        'scores': [{'knots': item['knots'], 'score': item['score'],
                    'rule': 'n*log(RSS/n) + (p + %s*K)*log(n)' % options['change_point_penalty']}
                   for item in selection['evaluations']],
        'services': services,
        'temperature_estimated_in_window': True,
        'status': '已选择',
    }
    column = fit['design']['temperature_column']
    variance = fit['covariance'][column][column] if fit['covariance'] is not None else None
    result['temperature'] = {
        'coefficient': fit['beta'][column],
        'interval': None if variance is None else [
            fit['beta'][column] - Z_95 * math.sqrt(max(variance, 0.0)),
            fit['beta'][column] + Z_95 * math.sqrt(max(variance, 0.0))],
        'reference': fit['design']['temperature_ref'],
        'estimated_in_window': True,
        'prior': None if prior is None else {
            'mean': prior[0], 'precision': prior[1],
            'source': '同 model 其他设备的窗口内估计，不共享观测'},
    }
    if float(result['temperature']['coefficient']) == 0.0 and variance is None:
        result['notes'].append('温度在设计矩阵中共线，未能单独估计')
    jumps = []
    for position, column in enumerate(fit['design']['jump_columns']):
        variance = fit['covariance'][column][column] if fit['covariance'] is not None else None
        half = Z_95 * math.sqrt(max(variance, 0.0)) if variance is not None else None
        jumps.append({
            'day': fit['design']['jump_days'][position],
            'jump': fit['beta'][column],
            'interval': None if half is None else [fit['beta'][column] - half,
                                                   fit['beta'][column] + half],
            'interpretation': '维护水平跳变，不计入退化趋势',
        })
    result['service_effects'] = jumps
    result['segments'] = segment_table(fit, fit['rows'], options)
    onset = _onset(fit, options)
    confirmation = _confirmation(rows, fit, onset, options)
    result['onset'] = onset
    result['confirmation'] = confirmation
    stability = bootstrap_stability(rows, services, selection, options, prior)
    result['stability'] = stability
    result['evidence'] = _evidence(rows, selection, stability, options, onset, confirmation)
    result['ambiguous_structure'] = _ambiguous(selection, options)
    if result['ambiguous_structure']:
        result['notes'].append('零拐点结构仅以小于 %.1f 的分差胜出，段数本身不确定'
                               % options['ambiguity_margin'])
    alternatives = [item for item in selection['evaluations']
                    if item['knots'] != chosen['knots']]
    if chosen['knots'] and alternatives:
        gap = min(item['score'] for item in alternatives) - chosen['score']
        if gap < options['ambiguity_margin']:
            result['notes'].append('存在评分接近的备选拐点结构，日期稳定性以重采样为准')
    return result


def _visible(records, cutoff):
    """只保留到 cutoff 为止已经可见、且已发生的观测。"""
    return [row for row in records
            if row['visible_at'] <= cutoff and row['day'] <= cutoff]


def _future_confirms(records, cutoff, options):
    """检查 cutoff 之后实际到达的数据是否支持持续退化。"""
    future = [row for row in records
              if row['day'] > cutoff and row['hours'] > 0 and row['visible_at'] > cutoff]
    future.sort(key=lambda row: (row['day'], row['observation_id']))
    if len(future) < options['min_confirmation_observations']:
        return {'resolved': False, 'confirmed': False,
                'reason': '之后数据不足，无法判定', 'observations': len(future)}
    window = list(future)
    services = service_events(window, options['min_side_days'])
    fit = fit_structure(window, [], services, options)
    if fit is None:
        return {'resolved': False, 'confirmed': False,
                'reason': '之后数据无法拟合', 'observations': len(window)}
    slopes = [item['slope'] for item in segment_table(fit, fit['rows'], options)]
    first, last = window[0], window[-1]
    # 用端点均值而不是单点，避免一条噪声把跨度很小的检验判死或判活。
    early = [row for row in window if row['day'] == first['day']]
    late = [row for row in window if row['day'] == last['day']]
    observed_drop = statistics.fmean(row['throughput'] for row in early) \
        - statistics.fmean(row['throughput'] for row in late)
    span = last['day'] - first['day']
    slope = min(slopes)
    expected_drop = -slope * span
    confirmed = (slope <= -options['degradation_threshold']
                 and expected_drop >= options['confirmation_min_drop']
                 and observed_drop >= 0.5 * options['confirmation_min_drop'])
    return {
        'resolved': True,
        'confirmed': bool(confirmed),
        'slope': slope,
        'span_days': span,
        'expected_drop': expected_drop,
        'observed_drop': observed_drop,
        'observations': len(window),
        'reason': None if confirmed else '之后数据未显示持续下降',
    }


def rolling_validation(records, options, reference=None, model_of=None):
    """历史滚动验证：用 visible_at 已到达的数据提出维护提示，再用之后数据核对。

    - 每个截点只用 ``visible_at<=cutoff`` 且 ``day<=cutoff`` 的记录。
    - 维护事件同样只保留 cutoff 之前已发生、已可见的，未来维护日期不进入过去模型。
    - 提示提出后，用 cutoff 之后实际到达的数据检验“持续退化”是否成立。
    - 检测延迟相对全量数据的参考起点计算；误报与无法判定分别计数。
    """
    by_asset = defaultdict(list)
    for row in records:
        by_asset[row['asset']].append(row)
    model_of = model_of or {asset: rows[0]['model'] for asset, rows in by_asset.items()}
    reference = reference or {}
    proposals = []
    totals = {'cutoffs': 0, 'alerts': 0, 'confirmed_alerts': 0, 'false_alerts': 0,
              'unresolved_alerts': 0, 'assets_with_detection': 0,
              'assets_with_false_alarm': 0, 'future_leaks': 0,
              'alerts_before_onset': 0}
    delays = []
    for asset in sorted(by_asset):
        rows = sorted(by_asset[asset], key=lambda row: (row['day'], row['observation_id']))
        observed_days = sorted({row['day'] for row in rows})
        if not observed_days:
            continue
        start = observed_days[0] + options['rolling_min_history']
        cutoffs = [day for day in observed_days if day >= start]
        asset_records = []
        first_detection = None
        had_false = False
        for cutoff in cutoffs:
            visible = _visible(rows, cutoff)
            usable_rows = usable(visible)
            excluded = [row for row in visible if row['hours'] <= 0]
            if len(usable_rows) < options['min_observations']:
                continue
            services = [event for event in service_events(visible, options['min_side_days'])
                        if event <= cutoff]
            # 滚动验证只需要“是否提出提示”；日期稳定性在主分析中重采样，
            # 内层重采样关掉，避免每个截点都付出完整 bootstrap 成本。
            fast = dict(options, bootstrap_replicates=0)
            diagnosis = diagnose_rows(usable_rows, model_of[asset], fast,
                                      None, excluded_rows=excluded)
            leaks = [event for event in diagnosis['services'] if event > cutoff]
            totals['future_leaks'] += len(leaks)
            totals['cutoffs'] += 1
            # 提示 = 窗口内已定位到起点且降幅达到确认下限；日期稳定性
            # 留给之后的真实数据核对，这正是“提前安排维护”要的时间提前量。
            alert = (diagnosis['onset'] is not None
                     and diagnosis['confirmation'] is not None
                     and diagnosis['confirmation']['confirmed'])
            record = {
                'cutoff': cutoff,
                'visible_observations': len(usable_rows),
                'services_used': services,
                'alert': bool(alert),
                'onset': None if diagnosis['onset'] is None else diagnosis['onset']['day'],
                'onset_type': None if diagnosis['onset'] is None else diagnosis['onset']['type'],
                'evidence': diagnosis['evidence'],
                'leaked_future_service_days': leaks,
            }
            if alert:
                totals['alerts'] += 1
                if diagnosis['onset']['type'] == 'before_window':
                    totals['alerts_before_onset'] += 1
                outcome = _future_confirms(rows, cutoff, options)
                record['outcome'] = outcome
                if not outcome['resolved']:
                    totals['unresolved_alerts'] += 1
                elif outcome['confirmed']:
                    totals['confirmed_alerts'] += 1
                    if first_detection is None:
                        first_detection = cutoff
                else:
                    totals['false_alerts'] += 1
                    had_false = True
            asset_records.append(record)
        if first_detection is not None:
            totals['assets_with_detection'] += 1
        if had_false:
            totals['assets_with_false_alarm'] += 1
        ref = reference.get(asset)
        delay = None
        if first_detection is not None and ref is not None:
            delay = first_detection - ref
            delays.append(delay)
        proposals.append({
            'asset': asset,
            'reference_onset': ref,
            'first_detection_day': first_detection,
            'detection_delay_days': delay,
            'cutoffs': asset_records,
        })
    resolved = totals['confirmed_alerts'] + totals['false_alerts']
    totals['resolved_alerts'] = resolved
    totals['alert_precision'] = (totals['confirmed_alerts'] / resolved
                                 if resolved else None)
    totals['false_alarm_rate'] = (totals['false_alerts'] / resolved
                                  if resolved else None)
    totals['detection_delay_median'] = statistics.median(delays) if delays else None
    totals['detected_after_onset'] = len(delays)
    return {'settings': {'rolling_min_history': options['rolling_min_history'],
                         'min_observations': options['min_observations'],
                         'degradation_threshold': options['degradation_threshold']},
            'totals': totals, 'assets': proposals,
            'note': '过去模型只使用 visible_at<=cutoff 且 day<=cutoff 的记录；'
                    '未来维护日期不进入过去模型。检测延迟以全量数据参考起点为基准；'
                    '无法判定的提示单独计数，不计入精确率分母。'}


def analyze(records, options=None, overrides=None):
    """按设备拟合，并在 model 层面共享先验强度；不拼接不同设备的观测。"""
    options = resolve_options({**(options or {}), **(overrides or {})})
    by_asset = defaultdict(list)
    for row in records:
        by_asset[row['asset']].append(row)
    for rows in by_asset.values():
        rows.sort(key=lambda row: (row['day'], row['observation_id']))
    model_of = {asset: rows[0]['model'] for asset, rows in by_asset.items()}
    usable_rows = {asset: usable(rows) for asset, rows in by_asset.items()}
    excluded_rows = {asset: [row for row in rows if row['hours'] <= 0]
                     for asset, rows in by_asset.items()}
    priors, first_pass = model_priors(usable_rows, options, model_of)
    assets = []
    for asset in sorted(by_asset):
        assets.append(diagnose_rows(usable_rows[asset], model_of[asset], options,
                                    _prior_tuple(priors.get(model_of[asset])),
                                    excluded_rows=excluded_rows[asset]))
    reference = {}
    for item in assets:
        if item['onset'] is not None:
            reference[item['asset']] = item['onset']['day']
    rolling = rolling_validation(records, options, reference, model_of)
    evidence_counts = Counter(item['evidence'] for item in assets)
    models = []
    for model in sorted({item['model'] for item in assets}):
        members = [item for item in assets if item['model'] == model]
        onsets = [item['onset']['day'] for item in members
                  if item['onset'] is not None and item['onset']['type'] == 'knot']
        models.append({
            'model': model,
            'assets': len(members),
            'temperature_prior': priors.get(model),
            'onset_days': onsets,
            'onset_median': statistics.median(onsets) if onsets else None,
            'evidence': dict(sorted(Counter(item['evidence'] for item in members).items())),
        })
    return {
        'product': PRODUCT,
        'schema_version': SCHEMA_VERSION,
        'unit': UNIT,
        'options': options,
        'assets': assets,
        'models': models,
        'rolling_validation': rolling,
        'summary': {
            'assets': len(assets),
            'evidence': dict(sorted(evidence_counts.items())),
            'degenerating': sorted(item['asset'] for item in assets
                                   if item['evidence'] == '持续退化'),
            'insufficient': sorted(item['asset'] for item in assets
                                   if item['evidence'] == '证据不足'),
        },
        'notes': [
            '每台设备单独拟合，model 层面只共享温度系数先验，不拼接观测。',
            'service_day 作为已知水平跳变，不作为趋势拐点候选。',
            '斜率区间是系数区间，不是拐点日期区间；日期范围来自重抽样稳定性。',
            '停机日（hours=0）不进入似然，也不把 throughput 记为零。',
        ],
    }


def _prior_tuple(prior):
    if prior is None or not prior.get('applied'):
        return None
    return (prior['mean'], prior['precision'])
