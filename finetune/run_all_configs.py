import sys
import subprocess
import glob
import requests
import getpass

api_key = getpass.getpass("Enter Lambda API key: ")

auth_header = f"Bearer {api_key}"

# Retrieve all Lambda Labs instances
instances_url = "https://cloud.lambda.ai/api/v1/instances"
instances_headers = {
    "accept": "application/json",
    "Authorization": auth_header
}
instances_response = requests.get(instances_url, headers=instances_headers)
instances = instances_response.json()["data"]

num_instances = len(instances)

if num_instances != 1:
    raise RuntimeError(f"Found {num_instances} running instances (expected 1).")

ip = instances[0]["ip"]
instance_id = instances[0]["id"]

print(f"Found instance {instance_id} with ip {ip}")

paths = glob.glob('./configs/*')
paths.sort()
print("Found paths:")
print("\n".join(paths))

for config_file in paths:
    print(f"Running classification on {config_file}")
    subprocess.run(['../.venv/bin/python', 'run_classification_remote.py', ip, "ubuntu", config_file], stderr=sys.stderr, stdout=sys.stdout)

# Shutdown remote cloud instance
url = "https://cloud.lambda.ai/api/v1/instance-operations/terminate"
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "Authorization": auth_header
}
data = {
    "instance_ids": [instance_id]
}

response = requests.post(url, headers=headers, json=data)
print(f"Shutdown request sent. Status code: {response.status_code}")
