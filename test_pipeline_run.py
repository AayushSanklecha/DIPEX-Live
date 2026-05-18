import requests

url = "http://localhost:8000/api/pipeline/simple-run"
data = {
    "source_kind": "database",
    "source_input": "postgresql://dipex:dipex_secret@postgres:5432/dipex?table=proposals"
}

print(f"Testing {url} with source_input={data['source_input']}...")
response = requests.post(url, data=data)
print(f"Status Code: {response.status_code}")

try:
    print(response.json())
except Exception as e:
    print(response.text)
