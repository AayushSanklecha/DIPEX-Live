
import requests, json

data = {
    'source_kind': 'live',
    'source_input': json.dumps({
        'brokers': 'kafka:29092',
        'topic': 'dipex_pipeline',
        'group_id': 'test-consumer',
        'max_messages': 20
    })
}

try:
    r = requests.post('http://localhost:8000/api/pipeline/simple-run', data=data)
    print('STATUS', r.status_code)
    print(r.text[:500])
except Exception as e:
    print('EXCEPTION', e)

