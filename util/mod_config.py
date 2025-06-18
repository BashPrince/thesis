import os
import json
from glob import glob

CONFIG_DIR = os.path.join(os.path.dirname(__file__), '../finetune/configs')

def swap_run_name(run_name):
    if run_name.endswith('gc_eval'):
        return run_name[:-7] + 'ct24_eval'
    elif run_name.endswith('ct24_eval'):
        return run_name[:-9] + 'gc_eval'
    return run_name

def swap_data_artifact(data_artifact):
    if data_artifact.startswith('ct24:'):
        return data_artifact.replace('ct24', 'general_claim_filtered', 1)
    elif data_artifact.startswith('general_claim_filtered:'):
        return data_artifact.replace('general_claim_filtered', 'ct24', 1)
    return data_artifact

def main():
    json_files = glob(os.path.join(CONFIG_DIR, '*.json'))
    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
        # ...existing code...
        if 'run_name' in data:
            data['run_name'] = swap_run_name(data['run_name'])
        if 'data_artifact' in data:
            data['data_artifact'] = swap_data_artifact(data['data_artifact'])
        # ...existing code...
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

if __name__ == '__main__':
    main()