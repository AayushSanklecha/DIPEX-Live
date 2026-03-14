
from api.routes.pipeline_run import _stream_cfg_from_input
import json

source_input = json.dumps({
    'brokers': 'kafka:29092',
    'topic': 'dipex_pipeline',
    'group_id': 'test-consumer',
    'max_messages': 20
})
conf = {'streaming': {}}
res = _stream_cfg_from_input(source_input, conf)
print(res)

