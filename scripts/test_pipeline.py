import requests
import time

print("1. Uploading test dataset...")
files = {
    'file': ('test.csv', 'id,age,score_val\n1,34,99.5\n2,28,88.0\n3,45,91.2\n4,22,76.5\n5,50,95.0\n6,38,89.5\n7,41,92.0\n8,29,81.5\n9,31,85.0\n10,25,79.0\n')
}
res_upload = requests.post('http://localhost:8000/api/ingest/', files=files)
run_id = res_upload.json()['run_id']
print(f"Uploaded! Run ID: {run_id}")

time.sleep(1)

print("2. Triggering pipeline...")
payload = {
    "run_id": run_id,
    "target_column": "score_val"
}
res_run = requests.post('http://localhost:8000/api/run/', json=payload)
print("Pipeline Run Result:")
# We just want to see the proposals part to verify the confidence scorer
run_data = res_run.json()

if "data" in run_data and "pipeline_result" in run_data["data"]:
    pr = run_data["data"]["pipeline_result"]
    if "analytics_result" in pr and pr["analytics_result"]:
        insights = pr["analytics_result"].get("insights", [])
        print("\n--- ML Confidence Scores ---")
        for i in insights:
            print(f"- {i.get('insight_type', 'Unknown')}: {i.get('confidence_score', 'N/A')} ({i.get('method', 'N/A')})")
    else:
        print("No analytics result found.")
else:
    print(run_data)
