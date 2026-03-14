import requests
import json
import time

def run_test():
    url = "http://127.0.0.1:8000/api/pipeline/simple-run"
    
    print("Test 1: JSONPlaceholder (Users)")
    data1 = {
        "source_kind": "api",
        "source_input": "https://jsonplaceholder.typicode.com/users",
        "skip_stages": "drift_detection,experience_memory,rl_update,causal_discovery"
    }
    res1 = requests.post(url, data=data1)
    print(res1.status_code)
    
    print("\nTest 2: REST Countries")
    data2 = {
        "source_kind": "api",
        "source_input": "https://restcountries.com/v3.1/all",
        "skip_stages": "drift_detection,experience_memory,rl_update,causal_discovery"
    }
    res2 = requests.post(url, data=data2)
    print(res2.status_code)
    try:
        j2 = res2.json()
        print(j2.get("final_result", {}).get("gate_decision"))
        if "detail" in j2:
            print("Error details:", j2["detail"])
    except Exception as e:
        print("Failed to parse JSON:", e)

if __name__ == "__main__":
    run_test()
