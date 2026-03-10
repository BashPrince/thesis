import pexpect
import subprocess
import sys
import os
import argparse

def ssh_execute_command(host_ip, username, command, ssh_port=22, expected_prompt=None):
    # Connect to the remote server using SSH
    ssh_command = f"ssh -p {ssh_port} {username}@{host_ip} {command}"

    # Spawn the process and initiate the SSH connection
    child = pexpect.spawn(ssh_command, encoding='utf-8')  # Ensure we use UTF-8 encoding

    # Log any output for debugging purposes (can be removed in production)
    child.logfile_read = sys.stdout

    if expected_prompt:
        for prompt in expected_prompt:
            child.expect(prompt, timeout=None)
            # If we detected the prompt huggingface prompt, respond with input
            child.sendline(input())  # Send the response to the remote script

    child.expect(pexpect.EOF, timeout=None)

    # Ensure the process is closed
    child.close()

def ssh_dir_exists(host_ip, username, remote_path, ssh_port=22):
    # Command to check if directory exists
    check_command = f"if [ -d '{remote_path}' ]; then echo 'EXISTS'; else echo 'NOT_EXISTS'; fi"
    ssh_command = f"ssh -p {ssh_port} {username}@{host_ip} {check_command}"

    child = pexpect.spawn(ssh_command, encoding='utf-8')
    child.expect(pexpect.EOF, timeout=None)
    output = child.before
    child.close()

    return 'NOT_EXISTS' not in output

def setup_workspace(host_ip, username, code_src_path, code_target_base_path, wandb_api_key, ssh_port=22):
    code_target_path = f"{code_target_base_path}/finetune"
    # Save an existing venv so we don't need to reinstall
    if ssh_dir_exists(host_ip, username, f"{code_target_path}/.venv", ssh_port):
        ssh_execute_command(host_ip, username, f"mv {code_target_path}/.venv {code_target_base_path}/.venv_backup", ssh_port)

    # Remove the existing directory on the remote server
    ssh_execute_command(host_ip, username, f"rm -rf {code_target_path}", ssh_port)

    # Transfer files using tar piped over SSH (no rsync dependency on remote)
    src_parent = os.path.dirname(os.path.normpath(code_src_path)) or '.'
    src_name = os.path.basename(os.path.normpath(code_src_path))
    transfer_command = (
        f"tar -czf - --exclude='.*' --exclude='secrets' -C {src_parent} {src_name}"
        f" | ssh -p {ssh_port} {username}@{host_ip}"
        f" 'mkdir -p ~/{code_target_base_path} && tar -xzf - -C ~/{code_target_base_path}'"
    )
    subprocess.run(transfer_command, shell=True, check=True)

    if ssh_dir_exists(host_ip, username, f"{code_target_base_path}/.venv_backup", ssh_port):
        # If a venv backup exists copy it back into the target path
        ssh_execute_command(host_ip, username, f"mv {code_target_base_path}/.venv_backup {code_target_path}/.venv", ssh_port)
    else:
        ssh_execute_command(host_ip, username, f"cd {code_target_path} && python3 -m venv .venv", ssh_port)
    # Always reinstall requirements to pick up any changes and ensure pip is up to date
    install_command = f"cd {code_target_path} && source .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt"
    ssh_execute_command(host_ip, username, install_command, ssh_port)

    # wandb login
    wandb_login_command = f'cd {code_target_path} && source .venv/bin/activate && wandb login {wandb_api_key}"'
    ssh_execute_command(host_ip, username, wandb_login_command, ssh_port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run remote classification training via SSH.")
    parser.add_argument("--host_ip", type=str, required=True, help="Remote host IP address")
    parser.add_argument("--username", type=str, required=True, help="Username for SSH")
    parser.add_argument("--ssh_port", type=int, default=22, help="SSH port (default: 22)")
    parser.add_argument("--config", type=str, required=True, help="Config file for training")
    parser.add_argument("--gpu_idx", type=int, required=True, help="Gpu index to run the script on")
    args = parser.parse_args()

    host_ip = args.host_ip
    username = args.username
    ssh_port = args.ssh_port
    config = args.config

    secrets_path = os.path.join(os.path.dirname(__file__), 'secrets', 'wandb_api_key')
    with open(secrets_path) as f:
        wandb_api_key = f.read().strip()

    code_src_path = "../finetune"
    code_target_base_path = f"finetune_{args.gpu_idx}"

    setup_workspace(host_ip, username, code_src_path, code_target_base_path, wandb_api_key, ssh_port)

    # Train on the remote server
    env_prefix = f"CUDA_VISIBLE_DEVICES={args.gpu_idx} "
    train_command = f"cd {code_target_base_path}/finetune && source .venv/bin/activate && {env_prefix}python run_classification.py {config}"
    ssh_execute_command(host_ip, username, train_command, ssh_port)
