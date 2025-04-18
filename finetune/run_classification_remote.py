import pexpect
import sys
import getpass  # Import getpass to securely prompt for sensitive input

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

def setup_workspace(host_ip, username, code_src_path, wandb_api_key):
    # Remove the existing directory on the remote server
    ssh_execute_command(host_ip, username, "rm -rf finetune")

    # Run the rsync command to copy files
    rsync_command = f"rsync -av --exclude='.*' {code_src_path} {username}@{host_ip}:~"
    pexpect.run(rsync_command, logfile=sys.stdout.buffer)

    # Use the ssh_execute_script function to install packages in a virtual environment
    install_command = "cd finetune && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    ssh_execute_command(host_ip, username, install_command)

    # wandb login
    wandb_login_command = f'cd finetune && source .venv/bin/activate && wandb login {wandb_api_key}"'
    ssh_execute_command(host_ip, username, wandb_login_command)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_classification_remote.py <host_ip> <username>")
        sys.exit(1)

    host_ip = sys.argv[1]
    username = sys.argv[2]

    # Prompt the user for the wandb_api_key
    wandb_api_key = getpass.getpass("Enter your wandb API key: ")

    code_src_path = "../finetune"

    setup_workspace(host_ip, username, code_src_path, wandb_api_key)

    # Train on the remote server
    train_command = f"cd finetune && python3 -m venv .venv && source .venv/bin/activate && python run_classification.py train_config.json"
    ssh_execute_command(host_ip, username, train_command)

