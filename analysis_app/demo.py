"""最小本地演示：跑通分段趋势、日期稳定性与滚动验证。

    python3 -m analysis_app.demo

只读取本地 ``data/observations.csv``，不联网。输出是给人看的中文摘要，
完整 JSON 请用 ``python3 -m analysis_app segments``。
"""

import time

from .io import load_records
from .segments import analyze

ASSETS = ('A00', 'A14')
STATUS_ORDER = ('持续退化', '证据不足', '退化（日期稳定性未评估）', '无趋势转折')


def _window(asset):
    window = asset['window']
    return f"d{window[0]}–d{window[1]}"


def main():
    started = time.time()
    records = load_records('data/observations.csv')
    result = analyze(records)
    elapsed = time.time() - started

    print(f"输入 {len(records)} 行，{result['summary']['assets']} 台设备，"
          f"耗时 {elapsed:.1f}s")
    print()
    print('证据分布：')
    for status in STATUS_ORDER:
        count = result['summary']['evidence'].get(status, 0)
        if count:
            print(f"  {status}: {count}")
    print()

    for asset in result['assets']:
        if asset['asset'] not in ASSETS:
            continue
        print(f"== {asset['asset']}（{asset['model']}，{_window(asset)}）："
              f"{asset['evidence']}")
        print(f"   停机日被排除 {asset['excluded_days']} 天；"
              f"维护事件 {asset['services'] or '无'}")
        for segment in asset['segments']:
            low, high = segment['slope_interval']
            print(f"   段 d{segment['start_day']}–d{segment['end_day']}: "
                  f"斜率 {segment['slope']:+.2f} "
                  f"[{low:+.2f}, {high:+.2f}] 件/小时/天")
        for jump in asset['service_effects']:
            print(f"   维护跳变 d{jump['day']}: {jump['jump']:+.2f} 件/小时"
                  '（不计入趋势）')
        rejected = [item for item in asset['candidates'] if not item['eligible']]
        if rejected:
            sample = rejected[len(rejected) // 2]
            print(f"   候选淘汰示例 d{sample['day']}: "
                  + '；'.join(sample['reasons']))
        stability = asset['stability']
        if stability['days']:
            knot = stability['days'][0]
            print(f"   拐点稳定性 {stability['status']}: 选中 d{knot['day']}，"
                  f"稳定范围 d{knot['range'][0]}–d{knot['range'][1]}，"
                  f"命中率 {knot['share']:.0%}"
                  f"（{stability['replicates']} 次重采样）")
        onset = asset['onset']
        if onset:
            print(f"   退化起点 {onset['day']}"
                  + ('（可能早于窗口）' if onset['type'] == 'before_window' else ''))
        print()

    rolling = result['rolling_validation']['totals']
    print('历史滚动验证（只用 visible_at 已到达的数据）：')
    print(f"   截点 {rolling['cutoffs']} 个，提示 {rolling['alerts']} 次；"
          f"可判定 {rolling['resolved_alerts']} 次中确认 {rolling['confirmed_alerts']}、"
          f"误报 {rolling['false_alerts']}")
    precision = rolling['alert_precision']
    delay = rolling['detection_delay_median']
    print(f"   提示精确率 {precision:.2f}" if precision is not None
          else '   提示精确率：暂无可判定样本')
    if delay is not None:
        print(f"   检测延迟中位数 {delay:g} 天（{rolling['detected_after_onset']} 台）")
    print(f"   未来维护日泄漏 {rolling['future_leaks']} 次")
    print()
    print('说明：斜率区间是系数区间，不是拐点日期区间；'
          '日期范围来自重采样稳定性。')


if __name__ == '__main__':
    main()
