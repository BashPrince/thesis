import sys
import subprocess
import glob
import requests
import getpass
import time
import os
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-k', '--keep', action='store_true', help='Keep the cloud instance running after jobs complete')
args = parser.parse_args()

api_key = getpass.getpass("Enter Lambda API key: ")

auth_header = f"Bearer {api_key}"

# Retrieve all Lambda Labs instances
instances_url = "https://cloud.lambda.ai/api/v1/instances"
instances_headers = {
    "accept": "application/json",
    "Authorization": auth_header
}
instances_response = requests.get(instances_url, headers=instances_headers)

if instances_response.status_code != 200:
    raise RuntimeError(f"Failed to retrieve instances: {instances_response.status_code} {instances_response.text}")

instance_json = instances_response.json()
instances = instance_json["data"]

num_instances = len(instances)

if num_instances != 1:
    raise RuntimeError(f"Found {num_instances} running instances (expected 1).")

ip = instances[0]["ip"]
instance_id = instances[0]["id"]
num_gpus = instances[0]["instance_type"]["specs"]["gpus"]

print(f"Found instance {instance_id} with ip {ip} and {num_gpus} gpus")

paths = glob.glob('./configs/*')
paths.sort()
print("Found paths:")
print("\n".join(paths))

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Track running processes and their assigned GPU indices
running = []
gpu_free = list(range(num_gpus))
config_iter = iter(paths)

def spawn_next():
    try:
        gpu_idx = gpu_free.pop(0)
    except IndexError:
        return  # No free GPU
    try:
        config_file = next(config_iter)
    except StopIteration:
        gpu_free.insert(0, gpu_idx)  # No more configs, free GPU
        return
    log_name = os.path.splitext(os.path.basename(config_file))[0] + ".log"
    log_path = os.path.join("logs", log_name)
    print(f"Running classification on {config_file} (GPU {gpu_idx}), logging to {log_path}")
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        [
            '../.venv/bin/python',
            'run_classification_remote.py',
            "--host_ip", ip,
            "--username", "ubuntu",
            "--config", config_file,
            "--gpu_idx", str(gpu_idx)
        ],
        stdout=log_fh,
        stderr=subprocess.STDOUT
    )
    running.append((proc, gpu_idx, log_fh))

# Start up to num_gpus processes
for _ in range(min(num_gpus, len(paths))):
    spawn_next()

# As processes finish, spawn new ones
while running:
    time.sleep(1)
    for i, (proc, gpu_idx, log_fh) in enumerate(running):
        ret = proc.poll()
        if ret is not None:
            # Process finished
            log_fh.close()
            gpu_free.append(gpu_idx)
            running.pop(i)
            spawn_next()
            break  # List changed, restart loop

# Shutdown remote cloud instance unless --keep is set
if not args.keep:
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
else:
    print("Skipping shutdown of cloud instance due to --keep flag.")
