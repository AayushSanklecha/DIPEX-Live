import requests
import json

url = "http://127.0.0.1:8000/api/pipeline/simple-run"
data = {
    "source_kind": "api",
    "source_input": "https://restcountries.com/v3.1/all?fields=name,capital,currencies,region,subregion",
    "skip_stages": "drift_detection,experience_memory,rl_update,causal_discovery"
}
res = requests.post(url, data=data)
print(res.status_code)
try:
    print(json.dumps(res.json(), indent=2))
except:
    print(res.text)
