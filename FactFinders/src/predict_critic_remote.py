import pexpect
import sys

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

def setup_workspace(host_ip, username, code_src_path, model_src_path):
    # Run the rsync command to copy files
    rsync_command = f"rsync -av --exclude='.*' {code_src_path} {username}@{host_ip}:critics"
    pexpect.run(rsync_command, logfile=sys.stdout.buffer)
    
    rsync_command = f"rsync -av {model_src_path} {username}@{host_ip}:critics/models"
    pexpect.run(rsync_command, logfile=sys.stdout.buffer)

    # Use the ssh_execute_script function to install packages in a virtual environment
    install_command = "cd critics/FactFinders && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    ssh_execute_command(host_ip, username, install_command)

    # HF login
    hf_login_command = 'cd critics/FactFinders && source .venv/bin/activate && huggingface-cli login"'
    ssh_execute_command(host_ip, username, hf_login_command, [r"Enter your token", r"Add token as git credential"])

def predict_and_copy_back_results(host_ip, username, model):
    # Run the prediction script on the remote server
    predict_command = f"cd critics/FactFinders && python3 -m venv .venv && source .venv/bin/activate && cd src && python predict.py --model_path '../../{model}'"
    ssh_execute_command(host_ip, username, predict_command)

    # Copy the prediction results back to the local machine
    rsync_command = f"rsync -av {username}@{host_ip}:critics/FactFinders/results/predict.csv ../results/predict_{model}.csv"
    pexpect.run(rsync_command, logfile=sys.stdout.buffer)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python predict_critic_remote.py <host_ip> <username> <model_src_path>")
        sys.exit(1)

    host_ip = sys.argv[1]
    username = sys.argv[2]
    model_src_path = sys.argv[3]

    code_src_path = "../../FactFinders"

    setup_workspace(host_ip, username, code_src_path, model_src_path)

    # predict with all models
    predict_and_copy_back_results(host_ip, username, 'gemma_7b')
    predict_and_copy_back_results(host_ip, username, 'mixtral_8x7b')
    predict_and_copy_back_results(host_ip, username, 'llama_3.1_8b')

