"""Synthetic series built for the boundary cases named in the task.

Every series is generated in-memory; no external files are needed.
"""

import math
import random


def make_rows(asset, model, days, throughput, hours, temperature, service_day=-1,
              visible_at=None, start_id=0):
    rows = []
    for position, (day, value, runtime, temp) in enumerate(
            zip(days, throughput, hours, temperature)):
        rows.append({
            'observation_id': f'{asset}-{start_id + position:04d}',
            'asset': asset,
            'model': model,
            'day': int(day),
            'throughput': float(value),
            'hours': float(runtime),
            'temperature': float(temp),
            'service_day': int(service_day),
            'visible_at': int(day if visible_at is None else visible_at[position]),
        })
    return rows


def changepoint_series(asset='A00', model='M0', start=100.0, slope_before=0.0,
                       slope_after=-2.0, knot=6, days=range(12), noise=0.6,
                       temperature_effect=0.5, seed=1, service_day=8):
    """A genuine trend break: flat then persistent decline."""
    rng = random.Random(seed)
    rows = []
    for day in days:
        trend = slope_before * day + (slope_after - slope_before) * max(0, day - knot)
        temp = 20.0 + 2.0 * math.sin(day * 0.9)
        value = start + trend + temperature_effect * (temp - 20.0) + rng.gauss(0.0, noise)
        rows.append((day, value, 8.0, temp))
    return make_rows(asset, model, [row[0] for row in rows], [row[1] for row in rows],
                     [row[2] for row in rows], [row[3] for row in rows],
                     service_day=service_day)


def flat_series(asset='A00', model='M0', start=100.0, days=range(12), noise=0.6,
                temperature_effect=0.5, seed=2, service_day=-1):
    """No trend at all: the correct answer is "no changepoint"."""
    rng = random.Random(seed)
    rows = []
    for day in days:
        temp = 20.0 + 2.0 * math.sin(day * 0.9)
        value = start + temperature_effect * (temp - 20.0) + rng.gauss(0.0, noise)
        rows.append((day, value, 8.0, temp))
    return make_rows(asset, model, [row[0] for row in rows], [row[1] for row in rows],
                     [row[2] for row in rows], [row[3] for row in rows],
                     service_day=service_day)


def maintenance_recovery_series(asset='A00', model='M0', start=100.0, service_day=6,
                                drop=-12.0, days=range(12), noise=0.6,
                                temperature_effect=0.5, seed=3):
    """Capacity drops before maintenance and recovers afterwards.

    The jump is a level change at a *known* maintenance day, not a trend
    change point; the detector must not call the recovery a degradation onset.
    """
    rng = random.Random(seed)
    rows = []
    for day in days:
        temp = 20.0 + 2.0 * math.sin(day * 0.9)
        level = start - (0.0 if day < service_day else drop)
        value = level + temperature_effect * (temp - 20.0) + rng.gauss(0.0, noise)
        rows.append((day, value, 8.0, temp))
    return make_rows(asset, model, [row[0] for row in rows], [row[1] for row in rows],
                     [row[2] for row in rows], [row[3] for row in rows],
                     service_day=service_day)


def irregular_series(asset='A00', model='M0', start=100.0, knot=8, slope_after=-2.0,
                     noise=0.5, temperature_effect=0.5, seed=4):
    """Non-uniform day spacing, including a long gap with no observations."""
    rng = random.Random(seed)
    days = [0, 2, 3, 5, 8, 9, 12, 16, 17, 21, 26, 30]
    rows = []
    for day in days:
        trend = slope_after * max(0, day - knot)
        temp = 20.0 + 2.0 * math.cos(day * 0.4)
        value = start + trend + temperature_effect * (temp - 20.0) + rng.gauss(0.0, noise)
        rows.append((day, value, 8.0, temp))
    return make_rows(asset, model, [row[0] for row in rows], [row[1] for row in rows],
                     [row[2] for row in rows], [row[3] for row in rows])


def outlier_series(asset='A00', model='M0', start=100.0, knot=6, slope_after=-2.0,
                   outlier_day=9, outlier_size=18.0, noise=0.5,
                   temperature_effect=0.5, seed=5):
    """One isolated extreme observation; it must not create a changepoint."""
    rng = random.Random(seed)
    rows = []
    for day in range(12):
        trend = slope_after * max(0, day - knot)
        temp = 20.0 + 2.0 * math.sin(day * 0.9)
        value = start + trend + temperature_effect * (temp - 20.0) + rng.gauss(0.0, noise)
        if day == outlier_day:
            value += outlier_size
        rows.append((day, value, 8.0, temp))
    return make_rows(asset, model, [row[0] for row in rows], [row[1] for row in rows],
                     [row[2] for row in rows], [row[3] for row in rows], service_day=-1)


def downtime_series(asset='A00', model='M0', start=100.0, knot=6, slope_after=-2.0,
                    downtime_days=(3, 4), noise=0.5, temperature_effect=0.5, seed=6):
    """Some days have hours=0; their throughput value must be ignored, not zeroed."""
    rng = random.Random(seed)
    rows = []
    for day in range(12):
        trend = slope_after * max(0, day - knot)
        temp = 20.0 + 2.0 * math.sin(day * 0.9)
        value = start + trend + temperature_effect * (temp - 20.0) + rng.gauss(0.0, noise)
        runtime = 0.0 if day in downtime_days else 8.0
        # during downtime throughput is unmeasured; put a deliberately wild value
        recorded = -999.0 if day in downtime_days else value
        rows.append((day, recorded, runtime, temp))
    return make_rows(asset, model, [row[0] for row in rows], [row[1] for row in rows],
                     [row[2] for row in rows], [row[3] for row in rows], service_day=-1)


def leap_series(asset='A00', model='M0', start=100.0, jump_day=7, jump=6.0,
                days=range(12), noise=0.5, temperature_effect=0.5, seed=7):
    """A pure step change with no trend: levels jump and then stay flat."""
    rng = random.Random(seed)
    rows = []
    for day in days:
        temp = 20.0 + 2.0 * math.sin(day * 0.9)
        level = start + (jump if day >= jump_day else 0.0)
        value = level + temperature_effect * (temp - 20.0) + rng.gauss(0.0, noise)
        rows.append((day, value, 8.0, temp))
    return make_rows(asset, model, [row[0] for row in rows], [row[1] for row in rows],
                     [row[2] for row in rows], [row[3] for row in rows], service_day=-1)
