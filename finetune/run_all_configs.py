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
parser.add_argument('--provider', choices=['lambda', 'runpod', 'auto'], default='auto',
                    help='Cloud provider to use (default: auto)')
args = parser.parse_args()

configs_dir = args.configs


def discover_lambda(api_key):
    """Discover a running Lambda Labs instance.
    Returns (ip, instance_id, num_gpus, username, ssh_port) or None if no instances.
    Raises if more than one instance is found.
    """
    url = "https://cloud.lambda.ai/api/v1/instances"
    headers = {"accept": "application/json", "Authorization": f"Bearer {api_key}"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise RuntimeError(f"Lambda API error: {response.status_code} {response.text}")
    instances = response.json()["data"]
    if len(instances) == 0:
        return None
    if len(instances) > 1:
        raise RuntimeError(f"Found {len(instances)} Lambda instances (expected 0 or 1).")
    inst = instances[0]
    return (inst["ip"], inst["id"], inst["instance_type"]["specs"]["gpus"], "ubuntu", 22)


def discover_runpod(api_key):
    """Discover a running Runpod pod.
    Returns (publicIp, podId, gpu_count, username, ssh_port) or None if no pods.
    Raises if more than one pod is found.
    """
    url = "https://rest.runpod.io/v1/pods?desiredStatus=RUNNING"
    headers = {"accept": "application/json", "Authorization": f"Bearer {api_key}"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise RuntimeError(f"Runpod API error: {response.status_code} {response.text}")
    pods = response.json()
    if len(pods) == 0:
        return None
    if len(pods) > 1:
        raise RuntimeError(f"Found {len(pods)} Runpod pods (expected 0 or 1).")
    pod = pods[0]
    ssh_port = pod["portMappings"]["22"]
    return (pod["publicIp"], pod["id"], pod["gpuCount"], "root", ssh_port)


def terminate_lambda(instance_id, api_key):
    url = "https://cloud.lambda.ai/api/v1/instance-operations/terminate"
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    response = requests.post(url, headers=headers, json={"instance_ids": [instance_id]})
    print(f"Lambda shutdown request sent. Status code: {response.status_code}")


def terminate_runpod(pod_id, api_key):
    url = f"https://rest.runpod.io/v1/pods/{pod_id}"
    headers = {"accept": "application/json", "Authorization": f"Bearer {api_key}"}
    response = requests.delete(url, headers=headers)
    print(f"Runpod shutdown request sent. Status code: {response.status_code}")


# Discover the running instance based on --provider
if args.provider in ('lambda', 'auto'):
    with open('secrets/lambda_api_key') as f:
        lambda_api_key = f.read().strip()
    lambda_result = discover_lambda(lambda_api_key)
else:
    lambda_api_key = None
    lambda_result = None

if args.provider in ('runpod', 'auto'):
    with open('secrets/runpod_api_key') as f:
        runpod_api_key = f.read().strip()
    runpod_result = discover_runpod(runpod_api_key)
else:
    runpod_api_key = None
    runpod_result = None

if args.provider == 'lambda':
    if lambda_result is None:
        raise RuntimeError("No Lambda instances found.")
    ip, instance_id, num_gpus, username, ssh_port = lambda_result
    terminate_fn = lambda: terminate_lambda(instance_id, lambda_api_key)
    print(f"Found Lambda instance {instance_id} at {ip} with {num_gpus} GPUs")
elif args.provider == 'runpod':
    if runpod_result is None:
        raise RuntimeError("No Runpod pods found.")
    ip, instance_id, num_gpus, username, ssh_port = runpod_result
    terminate_fn = lambda: terminate_runpod(instance_id, runpod_api_key)
    print(f"Found Runpod pod {instance_id} at {ip}:{ssh_port} with {num_gpus} GPUs")
else:  # auto
    found = []
    if lambda_result is not None:
        found.append(('lambda', lambda_result, lambda_api_key))
    if runpod_result is not None:
        found.append(('runpod', runpod_result, runpod_api_key))
    if len(found) == 0:
        raise RuntimeError("No running instances found on Lambda or Runpod.")
    if len(found) > 1:
        raise RuntimeError("Found running instances on multiple providers; use --provider to select one.")
    pname, result, pkey = found[0]
    ip, instance_id, num_gpus, username, ssh_port = result
    if pname == 'lambda':
        terminate_fn = lambda: terminate_lambda(instance_id, pkey)
        print(f"Found Lambda instance {instance_id} at {ip} with {num_gpus} GPUs")
    else:
        terminate_fn = lambda: terminate_runpod(instance_id, pkey)
        print(f"Found Runpod pod {instance_id} at {ip}:{ssh_port} with {num_gpus} GPUs")

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
                    "--username", username,
                    "--ssh_port", str(ssh_port),
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
    terminate_fn()
else:
    print("Skipping shutdown of cloud instance due to --keep flag.")
