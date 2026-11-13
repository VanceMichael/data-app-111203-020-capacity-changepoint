import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

from .io import load_records
from .segments import DEFAULT_OPTIONS, analyze, resolve_options
from .summary import summarize

OPTIONS = ('min-side-hours', 'min-side-days', 'min-segment-hours',
           'min-segment-days', 'min-observations', 'max-knots',
           'change-point-penalty', 'degradation-threshold',
           'confirmation-min-drop', 'stability-min-share',
           'bootstrap-replicates', 'bootstrap-seed', 'rolling-min-history',
           'service-window-days', 'robust-floor', 'min-side-gap-days')

KINDS = {'max-knots': int, 'min-side-days': int, 'min-segment-days': int,
         'min-observations': int, 'bootstrap-replicates': int,
         'bootstrap-seed': int, 'rolling-min-history': int,
         'service-window-days': int}


def _clean(value):
    """把 NaN/Inf 换成 None，保证 JSON 严格可读。"""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def save_json(path, result):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.analysis-', dir=destination.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            json.dump(_clean(result), handle, ensure_ascii=False, sort_keys=True,
                      allow_nan=False, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _overrides(namespace):
    overrides = {}
    for name in OPTIONS:
        value = getattr(namespace, name.replace('-', '_'), None)
        if value is None:
            continue
        overrides[name.replace('-', '_')] = value
    return overrides


def main():
    parser = argparse.ArgumentParser(description='本地观测数据分析')
    commands = parser.add_subparsers(dest='command', required=True)
    summary = commands.add_parser('summary', help='按业务维度核对观测口径')
    summary.add_argument('--input', type=Path, default=Path('data/observations.csv'))
    summary.add_argument('--output', type=Path)

    segments = commands.add_parser('segments', help='按设备分段趋势、拐点稳定性与滚动验证')
    segments.add_argument('--input', type=Path, default=Path('data/observations.csv'))
    segments.add_argument('--output', type=Path)
    segments.add_argument('--asset', action='append', default=None,
                          help='只分析指定设备，可重复')
    segments.add_argument('--no-bootstrap', action='store_true',
                          help='跳过日期稳定性重采样（更快，但日期状态不完整）')
    for name in OPTIONS:
        attribute = name.replace('-', '_')
        default = DEFAULT_OPTIONS.get(attribute)
        segments.add_argument('--' + name, dest=attribute, type=KINDS.get(name, float),
                              default=default, help='默认 %(default)s')
    args = parser.parse_args()
    try:
        records = load_records(args.input)
        if args.command == 'summary':
            result = summarize(records)
        else:
            overrides = _overrides(args)
            if args.asset:
                wanted = set(args.asset)
                records = [row for row in records if row['asset'] in wanted]
                if not records:
                    raise ValueError('指定的设备不在输入中：' + ','.join(sorted(wanted)))
            if args.no_bootstrap:
                overrides['bootstrap_replicates'] = 0
            resolve_options(overrides)
            result = analyze(records, overrides=overrides)
        if args.output:
            save_json(args.output, result)
        print(json.dumps(_clean(result), ensure_ascii=False, sort_keys=True, allow_nan=False))
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
