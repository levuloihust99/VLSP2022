#!/usr/bin/python3
import os
import re
import shutil


CHECKPOINT_DIR = "/home/lvloi/projects/vlsp-2022/submission-checkpoints"
CONFIG_FILE = "/home/lvloi/projects/vlsp-2022/seq2seq/tmp-vimds+abmusu/checkpoint-1617/config.json"

def main():
    cp_names = os.listdir(CHECKPOINT_DIR)
    for cp_name in cp_names:
        print(cp_name)
        cp_id = re.search(r"\d+", cp_name).group()
        cp_dir_name = f"checkpoint-{cp_id}"
        cp_dir_path = os.path.join(CHECKPOINT_DIR, cp_dir_name)
        if not os.path.exists(cp_dir_path):
            os.makedirs(cp_dir_path)
        cp_path = os.path.join(CHECKPOINT_DIR, f"pytorch_model_{cp_id}.bin")
        shutil.move(cp_path, os.path.join(cp_dir_path, "pytorch_model.bin"))
        shutil.copy(CONFIG_FILE, cp_dir_path)


if __name__ == "__main__":
    main()
