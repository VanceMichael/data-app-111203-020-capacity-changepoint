import math
import statistics
from collections import Counter


def describe(records):
    hours = math.fsum(row['hours'] for row in records)
    return {'runtime_hours': hours, 'estimated_output': math.fsum(row['throughput'] * row['hours'] for row in records), 'assets': len({row['asset'] for row in records})}
