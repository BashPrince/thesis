import pexpect
import sys
import getpass  # Import getpass to securely prompt for sensitive input
import os  # Import os to access environment variables
import argparse  # Add argparse for argument parsing

def ssh_execute_command(host_ip, username, command, expected_prompt=None):
    # Connect to the remote server using SSH
    ssh_command = f"ssh {username}@{host_ip} {command}"

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

def ssh_dir_exists(host_ip, username, remote_path):
    # Command to check if directory exists
    check_command = f"if [ -d '{remote_path}' ]; then echo 'EXISTS'; else echo 'NOT_EXISTS'; fi"
    ssh_command = f"ssh {username}@{host_ip} {check_command}"

    child = pexpect.spawn(ssh_command, encoding='utf-8')
    child.expect(pexpect.EOF, timeout=None)
    output = child.before
    child.close()

    return 'NOT_EXISTS' not in output

def setup_workspace(host_ip, username, code_src_path, code_target_base_path, wandb_api_key):
    code_target_path = f"{code_target_base_path}/finetune"
    # Save an existing venv so we don't need to reinstall
    if ssh_dir_exists(host_ip, username, f"{code_target_path}/.venv"):
        ssh_execute_command(host_ip, username, f"mv {code_target_path}/.venv {code_target_base_path}/.venv_backup")

    # Remove the existing directory on the remote server
    ssh_execute_command(host_ip, username, f"rm -rf {code_target_path}")

    # Run the rsync command to copy files
    rsync_command = f"rsync -av --exclude='.*' {code_src_path} {username}@{host_ip}:~/{code_target_base_path}"
    pexpect.run(rsync_command, logfile=sys.stdout.buffer)

    if ssh_dir_exists(host_ip, username, f"{code_target_base_path}/.venv_backup"):
        # If a venv backup exists copy it back into the target path
        ssh_execute_command(host_ip, username, f"mv {code_target_base_path}/.venv_backup {code_target_path}/.venv")
    else:
        # Use the ssh_execute_script function to install packages in a virtual environment
        install_command = f"cd {code_target_path} && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
        ssh_execute_command(host_ip, username, install_command)

    # wandb login
    wandb_login_command = f'cd {code_target_path} && source .venv/bin/activate && wandb login {wandb_api_key}"'
    ssh_execute_command(host_ip, username, wandb_login_command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run remote classification training via SSH.")
    parser.add_argument("--host_ip", type=str, required=True, help="Remote host IP address")
    parser.add_argument("--username", type=str, required=True, help="Username for SSH")
    parser.add_argument("--config", type=str, required=True, help="Config file for training")
    parser.add_argument("--gpu_idx", type=int, required=True, help="Gpu index to run the script on")
    args = parser.parse_args()

    host_ip = args.host_ip
    username = args.username
    config = args.config

    # Retrieve wandb_api_key from environment variable or prompt the user
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        wandb_api_key = getpass.getpass("Enter your wandb API key: ")

    code_src_path = "../finetune"
    code_target_base_path = f"finetune_{args.gpu_idx}"

    setup_workspace(host_ip, username, code_src_path, code_target_base_path, wandb_api_key)

    # Train on the remote server
    env_prefix = f"CUDA_VISIBLE_DEVICES={args.gpu_idx} "
    train_command = f"cd {code_target_base_path}/finetune && python3 -m venv .venv && source .venv/bin/activate && {env_prefix}python run_classification.py {config}"
    ssh_execute_command(host_ip, username, train_command)

