import sys
import subprocess
import glob
import requests
import time
import os
import json
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-k', '--keep', action='store_true', help='Keep the cloud instance running after jobs complete')
parser.add_argument('--configs', default='./configs', help='Directory containing config JSON files (default: ./configs)')
args = parser.parse_args()

configs_dir = args.configs

with open('secrets/lambda_api_key') as f:
    api_key = f.read().strip()

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

# Load configs (all .json files in configs_dir except dependencies.json)
all_paths = sorted(glob.glob(os.path.join(configs_dir, '*.json')))
paths = [p for p in all_paths if os.path.basename(p) != 'dependencies.json']
print("Found configs:")
print("\n".join(paths))

# Load dependency map: { config_basename -> [dep_basename, ...] }
# Only classify configs have entries; contrastive configs have no deps.
dep_file = os.path.join(configs_dir, 'dependencies.json')
dependencies = {}
if os.path.exists(dep_file):
    with open(dep_file) as f:
        dependencies = json.load(f)
    print(f"Loaded {len(dependencies)} dependency rule(s) from {dep_file}")

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Scheduling state
pending   = list(paths)   # configs not yet started, preserved in sorted order
running   = []            # list of (proc, gpu_idx, log_fh, config_basename)
completed = set()         # basenames of successfully finished configs
failed    = set()         # basenames of failed or skipped configs
gpu_free  = list(range(num_gpus))


def deps_satisfied(config_path):
    """All declared dependencies have completed successfully."""
    name = os.path.basename(config_path)
    return all(dep in completed for dep in dependencies.get(name, []))


def deps_failed(config_path):
    """At least one declared dependency has failed or been skipped."""
    name = os.path.basename(config_path)
    return any(dep in failed for dep in dependencies.get(name, []))


def spawn_next():
    """Try to start the next pending config whose dependencies are met.

    Iterates pending in order, skipping (and marking failed) any config whose
    dependency has failed, and stopping at the first config that is ready.
    Returns True if a job was spawned, False otherwise.
    """
    if not gpu_free:
        return False

    for config_file in list(pending):
        if deps_failed(config_file):
            name = os.path.basename(config_file)
            print(f"Skipping {config_file}: a dependency failed or was skipped")
            pending.remove(config_file)
            failed.add(name)
            continue  # Check the next pending config

        if deps_satisfied(config_file):
            gpu_idx = gpu_free.pop(0)
            pending.remove(config_file)
            name     = os.path.basename(config_file)
            log_path = os.path.join("logs", os.path.splitext(name)[0] + ".log")
            print(f"Running {config_file} on GPU {gpu_idx}, logging to {log_path}")
            log_fh = open(log_path, "w")
            proc = subprocess.Popen(
                [
                    '../.venv/bin/python',
                    'run_classification_remote.py',
                    "--host_ip",  ip,
                    "--username", "ubuntu",
                    "--config",   config_file,
                    "--gpu_idx",  str(gpu_idx),
                ],
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )
            running.append((proc, gpu_idx, log_fh, name))
            return True

    return False  # Nothing ready to run right now


def spawn_all_ready():
    """Fill all free GPUs with ready configs."""
    while gpu_free and pending:
        if not spawn_next():
            break  # Nothing ready; remaining configs are waiting on dependencies


# Start as many jobs as we have GPUs
spawn_all_ready()

# Main scheduling loop: keep going while jobs are running or configs are pending
while running or pending:
    time.sleep(1)
    for i, (proc, gpu_idx, log_fh, name) in enumerate(running):
        ret = proc.poll()
        if ret is not None:
            log_fh.close()
            gpu_free.append(gpu_idx)
            running.pop(i)

            if ret == 0:
                completed.add(name)
                print(f"{name} finished successfully "
                      f"({len(completed)} done, {len(pending)} pending, {len(running)} running)")
            else:
                failed.add(name)
                print(f"{name} FAILED with exit code {ret} "
                      f"({len(failed)} failed, {len(pending)} pending, {len(running)} running)")

            # A GPU is now free and a job may have unblocked dependencies
            spawn_all_ready()
            break  # Restart the for-loop (running list changed)

if failed:
    print(f"\nFinished with {len(failed)} failure(s): {sorted(failed)}")
else:
    print(f"\nAll {len(completed)} configs completed successfully")

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
