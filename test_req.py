
import requests, json
data = {
    'source_kind': 'live',
    'source_input': json.dumps({
        'brokers': 'kafka:29092',
        'topic': 'dipex_pipeline',
        'max_messages': 20,
        'group_id': 'test-consumer'
    })
}
r = requests.post('http://localhost:8000/api/pipeline/run', data=data)
print(r.status_code)
print(r.text[:500])

