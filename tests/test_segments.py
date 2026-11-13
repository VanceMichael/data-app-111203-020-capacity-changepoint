import copy
import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from analysis_app.cli import save_json
from analysis_app.io import load_records
from analysis_app.linalg import independent_columns, inverse, solve
from analysis_app.segments import (analyze, bootstrap_stability, eligible_knots,
                                   fit_structure, flag_outliers, resolve_options,
                                   segment_table, select_structure, service_events,
                                   usable)
from tests import series


def diagnose(rows, **overrides):
    return analyze(rows, overrides=overrides)['assets'][0]


class LinearAlgebraTest(unittest.TestCase):
    def test_solve_and_inverse_round_trip(self):
        matrix = [[4.0, 1.0], [1.0, 3.0]]
        beta = solve(matrix, [1.0, 2.0])
        self.assertAlmostEqual(matrix[0][0] * beta[0] + matrix[0][1] * beta[1], 1.0)
        product = inverse(matrix)
        self.assertAlmostEqual(product[0][0] * matrix[0][0] + product[0][1] * matrix[1][0], 1.0)

    def test_singular_system_returns_none(self):
        self.assertIsNone(solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0]))

    def test_collinear_column_is_dropped(self):
        kept = independent_columns([[1.0, 2.0], [2.0, 4.0], [1.0, 2.0]])
        self.assertEqual(kept, [0])


class SampleHandlingTest(unittest.TestCase):
    def test_downtime_is_not_zero_capacity(self):
        rows = series.downtime_series()
        kept = usable(rows)
        self.assertNotIn(-999.0, [row['throughput'] for row in kept])
        self.assertEqual(len(kept), len(rows) - 2)
        result = diagnose(rows)
        self.assertEqual(result['excluded_days'], 2)
        self.assertEqual(result['evidence'], '持续退化')

    def test_downtime_only_is_insufficient(self):
        rows = [dict(row, hours=0.0) for row in series.changepoint_series()]
        result = diagnose(rows)
        self.assertEqual(result['evidence'], '样本不足')
        self.assertEqual(result['usable_observations'], 0)

    def test_empty_input(self):
        result = analyze([])
        self.assertEqual(result['assets'], [])
        self.assertEqual(result['summary']['assets'], 0)


class ChangePointTest(unittest.TestCase):
    def test_genuine_break_is_found_near_truth(self):
        result = diagnose(series.changepoint_series(knot=6, slope_after=-2.0))
        self.assertEqual(result['evidence'], '持续退化')
        self.assertEqual(result['model_selection']['chosen_knots'], [6])
        self.assertEqual(result['onset']['day'], 6)
        slopes = [item['slope'] for item in result['segments']]
        self.assertAlmostEqual(slopes[0], 0.0, delta=0.6)
        self.assertAlmostEqual(slopes[1], -2.0, delta=0.6)

    def test_irregular_sampling(self):
        result = diagnose(series.irregular_series(knot=8))
        self.assertEqual(result['evidence'], '持续退化')
        self.assertEqual(result['onset']['day'], 8)

    def test_single_outlier_does_not_become_a_break(self):
        clean = diagnose(series.changepoint_series(knot=6, slope_after=-2.0, seed=5))
        dirty = diagnose(series.outlier_series(outlier_day=9, outlier_size=18.0))
        self.assertEqual(clean['model_selection']['chosen_knots'], [6])
        # 单次离群不应把拐点拉到离群日；仍应定位在真实转折附近。
        self.assertEqual(dirty['model_selection']['chosen_knots'], [6])
        self.assertNotEqual(dirty['onset']['day'], 9)

    def test_flat_series_has_no_break(self):
        result = diagnose(series.flat_series(seed=2))
        self.assertEqual(result['model_selection']['chosen_knots'], [])
        self.assertEqual(result['evidence'], '无趋势转折')
        self.assertIsNone(result['onset'])

    def test_pure_level_jump_is_not_a_trend_turn(self):
        result = diagnose(series.leap_series(jump_day=7))
        # 台阶本身不是“持续退化”；不能把跳变后的高平台当成趋势转折。
        self.assertNotEqual(result['evidence'], '持续退化')

    def test_maintenance_recovery_is_a_jump_not_an_onset(self):
        result = diagnose(series.maintenance_recovery_series(service_day=6, drop=-12.0))
        self.assertEqual(result['evidence'], '无趋势转折')
        self.assertIsNone(result['onset'])
        self.assertEqual([item['day'] for item in result['service_effects']], [6])
        reasons = {item['day']: item['reasons'] for item in result['candidates']}
        self.assertTrue(any('维护' in text for text in reasons[6]))

    def test_service_day_never_enters_knot_candidates(self):
        rows = series.changepoint_series(service_day=5)
        kept = service_events(rows, min_side_days=3)
        self.assertEqual(kept, [5])
        eligible, rejected = eligible_knots(usable(rows), kept, resolve_options())
        self.assertNotIn(5, [item['day'] for item in eligible])
        self.assertIn(5, [item['day'] for item in rejected])

    def test_temperature_is_estimated_inside_each_window(self):
        rows = series.changepoint_series(temperature_effect=1.0)
        options = resolve_options()
        first = fit_structure(usable(rows)[:6], [], [], options)
        second = fit_structure(usable(rows)[6:], [], [], options)
        full = fit_structure(usable(rows), [], [], options)
        column = first['design']['temperature_column']
        self.assertNotAlmostEqual(first['beta'][column], second['beta'][column], places=3)
        # 每个窗口的温度参照点不同，说明是在窗口内估计而不是复用全局系数。
        self.assertNotAlmostEqual(first['design']['temperature_ref'],
                                  full['design']['temperature_ref'], places=3)

    def test_segment_slopes_match_model(self):
        rows = series.changepoint_series(knot=6, slope_before=0.3, slope_after=-1.5,
                                         noise=0.2)
        selection = select_structure(usable(rows), service_events(rows), resolve_options())
        table = segment_table(selection['chosen']['fit'], usable(rows), resolve_options())
        self.assertEqual([item['start_day'] for item in table], [0, 6])
        self.assertAlmostEqual(table[0]['slope'], 0.3, delta=0.4)
        self.assertAlmostEqual(table[1]['slope'], -1.5, delta=0.4)
        self.assertIn('不是拐点日期区间', table[0]['interval_note'])


class SelectionTest(unittest.TestCase):
    def test_complexity_penalty_selects_fewer_segments(self):
        rows = series.flat_series(seed=11)
        strict = select_structure(usable(rows), [], resolve_options({'max_knots': 2}))
        heavy = select_structure(usable(rows), [], resolve_options({'max_knots': 2,
                                                                   'change_point_penalty': 40.0}))
        self.assertLessEqual(len(heavy['chosen']['knots']), len(strict['chosen']['knots']))
        self.assertEqual(heavy['chosen']['knots'], [])

    def test_candidate_rejection_reasons_are_reported(self):
        rows = series.changepoint_series()
        result = diagnose(rows)
        self.assertTrue(result['candidates'])
        rejected = [item for item in result['candidates'] if not item['eligible']]
        self.assertTrue(all(item['reasons'] for item in rejected))
        self.assertTrue(all(item['eligible'] or item['reasons']
                            for item in result['candidates']))

    def test_insufficient_evidence_state(self):
        rows = series.changepoint_series(days=range(5))
        result = diagnose(rows)
        self.assertEqual(result['evidence'], '样本不足')

    def test_spike_detection_flags_only_the_outlier(self):
        clean = usable(series.changepoint_series())
        dirty = usable(series.outlier_series(outlier_day=9, outlier_size=18.0))
        self.assertEqual(flag_outliers(clean, resolve_options()), [])
        flagged = [dirty[index]['day'] for index in flag_outliers(dirty, resolve_options())]
        self.assertEqual(flagged, [9])

    def test_outliers_are_reported_and_excluded(self):
        result = diagnose(series.outlier_series(outlier_day=9, outlier_size=18.0))
        self.assertEqual([item['day'] for item in result['outliers']], [9])
        self.assertEqual(result['model_selection']['chosen_knots'], [6])

    def test_structural_ambiguity_is_not_declared_as_no_trend(self):
        rows = series.changepoint_series(knot=6, slope_after=-0.8, noise=1.2, seed=8)
        result = diagnose(rows)
        if result['ambiguous_structure']:
            self.assertEqual(result['evidence'], '证据不足')
            self.assertIn('段数本身不确定', ' '.join(result['notes']))

    def test_stability_reports_insufficient_when_dates_scatter(self):
        # 纯噪声没有真实拐点，重采样日期必然发散。
        rows = series.flat_series(seed=21)
        result = diagnose(rows, bootstrap_replicates=30)
        if result['model_selection']['chosen_knots']:
            self.assertIn(result['stability']['status'], ('证据不足', '基本稳定'))


class BootstrapTest(unittest.TestCase):
    def test_stability_gives_date_range_not_regression_interval(self):
        result = diagnose(series.changepoint_series(knot=6, noise=0.4),
                          bootstrap_replicates=40)
        stability = result['stability']
        self.assertEqual(stability['replicates'], 40)
        self.assertTrue(stability['days'])
        knot = stability['days'][0]
        self.assertIsNotNone(knot['range'])
        self.assertLessEqual(knot['range'][0], knot['range'][1])
        self.assertIn('share', knot)
        self.assertIn('斜率系数', result['segments'][0]['interval_note'])

    def test_zero_replicates_is_explicit(self):
        result = diagnose(series.changepoint_series(), bootstrap_replicates=0)
        self.assertEqual(result['stability']['replicates'], 0)
        # 跳过重采样时不能把“未评估”读成“证据不足”。
        self.assertEqual(result['stability']['status'], '未评估')
        self.assertEqual(result['evidence'], '退化（日期稳定性未评估）')

    def test_bootstrap_is_deterministic(self):
        rows = series.changepoint_series()
        first = diagnose(rows, bootstrap_replicates=10)
        second = diagnose(rows, bootstrap_replicates=10)
        self.assertEqual(first['stability'], second['stability'])


class RollingValidationTest(unittest.TestCase):
    def test_past_models_only_use_visible_data(self):
        rows = series.changepoint_series()
        result = analyze(rows, overrides={'rolling_min_history': 5,
                                          'bootstrap_replicates': 0})
        rolling = result['rolling_validation']
        self.assertEqual(rolling['totals']['future_leaks'], 0)
        for asset in rolling['assets']:
            for record in asset['cutoffs']:
                self.assertLessEqual(record['cutoff'], 11)
                for event in record['services_used']:
                    self.assertLessEqual(event, record['cutoff'])

    def test_future_service_day_is_hidden_before_it_happens(self):
        rows = series.changepoint_series(service_day=8)
        result = analyze(rows, overrides={'rolling_min_history': 4,
                                          'bootstrap_replicates': 0})
        asset = result['rolling_validation']['assets'][0]
        early = [record for record in asset['cutoffs'] if record['cutoff'] < 8]
        self.assertTrue(early)
        for record in early:
            self.assertNotIn(8, record['services_used'])

    def test_detection_delay_and_false_alarms_are_reported(self):
        rows = series.changepoint_series()
        result = analyze(rows, overrides={'rolling_min_history': 5,
                                          'bootstrap_replicates': 0})
        totals = result['rolling_validation']['totals']
        for key in ('alerts', 'confirmed_alerts', 'false_alerts',
                    'unresolved_alerts', 'alert_precision', 'false_alarm_rate'):
            self.assertIn(key, totals)
        self.assertIn('detection_delay_days',
                      result['rolling_validation']['assets'][0])

    def test_flat_series_should_not_alert_early(self):
        rows = series.flat_series(seed=2)
        result = analyze(rows, overrides={'rolling_min_history': 5,
                                          'bootstrap_replicates': 0})
        asset = result['rolling_validation']['assets'][0]
        confirmed = [record for record in asset['cutoffs']
                     if record.get('outcome', {}).get('confirmed')]
        self.assertEqual(confirmed, [])


class IsolationTest(unittest.TestCase):
    def test_assets_are_not_concatenated(self):
        rows = series.changepoint_series(asset='A', start=100.0)
        rows += series.flat_series(asset='B', start=300.0, seed=9)
        result = analyze(rows, overrides={'bootstrap_replicates': 0})
        by_asset = {item['asset']: item for item in result['assets']}
        self.assertEqual(by_asset['A']['window'], [0, 11])
        self.assertEqual(by_asset['B']['window'], [0, 11])
        self.assertEqual(by_asset['A']['model_selection']['chosen_knots'], [6])
        self.assertEqual(by_asset['B']['model_selection']['chosen_knots'], [])
        # 高起点设备不会把低起点设备的斜率带偏。
        self.assertAlmostEqual(by_asset['A']['segments'][1]['slope'], -2.0, delta=0.6)

    def test_level_offsets_do_not_change_slopes(self):
        low = series.changepoint_series(asset='L', start=100.0, seed=11)
        high = series.changepoint_series(asset='H', start=500.0, seed=11)
        result = analyze(low + high, overrides={'bootstrap_replicates': 0})
        by_asset = {item['asset']: item for item in result['assets']}
        self.assertEqual(by_asset['L']['model_selection']['chosen_knots'], [6])
        self.assertEqual(by_asset['H']['model_selection']['chosen_knots'], [6])
        self.assertEqual([round(item['slope'], 6) for item in by_asset['L']['segments']],
                         [round(item['slope'], 6) for item in by_asset['H']['segments']])

    def test_model_prior_does_not_share_observations(self):
        first = series.changepoint_series(asset='A', model='M0', seed=1)
        second = series.changepoint_series(asset='B', model='M0', start=200.0, seed=2)
        third = series.changepoint_series(asset='C', model='M1', start=50.0, seed=3)
        rows = first + second + third
        result = analyze(rows, overrides={'bootstrap_replicates': 0})
        models = {item['model']: item for item in result['models']}
        self.assertEqual(models['M0']['assets'], 2)
        self.assertEqual(models['M1']['assets'], 1)
        self.assertEqual(models['M1']['temperature_prior']['method'],
                         '单一设备，不施加组先验')
        self.assertFalse(models['M1']['temperature_prior']['applied'])
        priors = {item['asset']: item['temperature']['prior']
                  for item in result['assets']}
        self.assertIsNotNone(priors['A'])
        self.assertIsNotNone(priors['B'])
        self.assertEqual(priors['A']['source'], '同 model 其他设备的窗口内估计，不共享观测')

    def test_input_order_does_not_change_result(self):
        rows = series.changepoint_series()
        shuffled = copy.deepcopy(rows)
        random.Random(7).shuffle(shuffled)
        first = analyze(rows, overrides={'bootstrap_replicates': 0})
        second = analyze(shuffled, overrides={'bootstrap_replicates': 0})
        self.assertEqual(first, second)


class OptionTest(unittest.TestCase):
    def test_unknown_option_rejected(self):
        with self.assertRaisesRegex(ValueError, '未知分析选项'):
            resolve_options({'mystery': 1})

    def test_option_bounds(self):
        with self.assertRaises(ValueError):
            resolve_options({'max_knots': 99})
        with self.assertRaises(ValueError):
            resolve_options({'stability_min_share': 0.0})
        with self.assertRaises(ValueError):
            resolve_options({'min_side_hours': -1.0})

    def test_bad_type_rejected(self):
        with self.assertRaises(ValueError):
            resolve_options({'bootstrap_replicates': -3})
        with self.assertRaises(ValueError):
            resolve_options({'min_side_hours': float('nan')})


class CliTest(unittest.TestCase):
    def test_segments_cli_matches_library(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            run = subprocess.run(
                [sys.executable, '-m', 'analysis_app', 'segments',
                 '--asset', 'A00', '--no-bootstrap', '--output', str(path)],
                text=True, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            payload = json.loads(run.stdout)
            self.assertEqual(payload['assets'][0]['asset'], 'A00')
            self.assertEqual(json.loads(path.read_text()), payload)

    def test_segments_cli_unknown_asset(self):
        run = subprocess.run([sys.executable, '-m', 'analysis_app', 'segments',
                              '--asset', 'missing'], text=True, capture_output=True)
        self.assertEqual(run.returncode, 2)
        self.assertIn('指定的设备', run.stderr)

    def test_json_has_no_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            run = subprocess.run([sys.executable, '-m', 'analysis_app', 'segments',
                                  '--no-bootstrap', '--output', str(path)],
                                 text=True, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            text = path.read_text()
            self.assertNotIn('NaN', text)
            self.assertNotIn('Infinity', text)


if __name__ == '__main__':
    unittest.main()
