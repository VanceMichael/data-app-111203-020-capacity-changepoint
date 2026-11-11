FIELDS = {'observation_id': 'str', 'asset': 'str', 'model': 'str', 'day': 'int', 'throughput': 'float', 'hours': 'float', 'temperature': 'float', 'service_day': 'int', 'visible_at': 'int'}
REQUIRED = ['observation_id', 'asset', 'model', 'throughput']
ENTITY = 'asset'
GROUP = 'model'
MEASURE = 'throughput'
PRODUCT = 'capacity-changepoint'
UNIT = '件每小时'
