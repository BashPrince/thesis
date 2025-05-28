import pexpect
import sys
import getpass  # Import getpass to securely prompt for sensitive input
import os  # Import os to access environment variables

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

def setup_workspace(host_ip, username, code_src_path, wandb_api_key):
    # Save an existing venv so we don't need to reinstall
    if ssh_dir_exists(host_ip, username, "analysis/.venv"):
        ssh_execute_command(host_ip, username, "mv analysis/.venv .venv_backup")

    # Remove the existing directory on the remote server
    ssh_execute_command(host_ip, username, "rm -rf analysis")

    # Run the rsync command to copy code
    rsync_command = f"rsync -av --exclude='.*' {code_src_path} {username}@{host_ip}:~"
    pexpect.run(rsync_command, logfile=sys.stdout.buffer)

    if ssh_dir_exists(host_ip, username, ".venv_backup"):
        # If a venv backup exists copy it back into analysis
        ssh_execute_command(host_ip, username, "mv .venv_backup analysis/.venv")
    else:
        # Use the ssh_execute_script function to install packages in a virtual environment
        install_command = "cd analysis && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
        ssh_execute_command(host_ip, username, install_command)

    # wandb login
    wandb_login_command = f'cd analysis && source .venv/bin/activate && wandb login {wandb_api_key}"'
    ssh_execute_command(host_ip, username, wandb_login_command)


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python run_flair_tagging_remote.py <host_ip> <username> <input_artifact> <output_artifact> <split>")
        sys.exit(1)

    host_ip = sys.argv[1]
    username = sys.argv[2]
    input_artifact = sys.argv[3]
    output_artifact = sys.argv[4]
    split = sys.argv[5]

    # Retrieve wandb_api_key from environment variable or prompt the user
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        wandb_api_key = getpass.getpass("Enter your wandb API key: ")

    code_src_path = "../analysis"

    setup_workspace(host_ip, username, code_src_path, wandb_api_key)

    # Train on the remote server
    tag_command = f"cd analysis && python3 -m venv .venv && source .venv/bin/activate && nohup python run_flair_tagging.py --input {input_artifact} --output {output_artifact} --split {split}"
    ssh_execute_command(host_ip, username, tag_command)

