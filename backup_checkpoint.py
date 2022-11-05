#!/usr/bin/python3

import os
import subprocess
import signal
import time


CHECKPOINT_DIR = "/home/lvloi/projects/vlsp-2022/seq2seq/tmp"
CHECKPOINT_STATE_FILE = "/home/lvloi/projects/vlsp-2022/backup_state.txt"


def signal_handler(signalnum, stackframe):
    print("\nOne subprocess terminated.\r")
    if not hasattr(signal_handler, 'num_terminated_procs'):
        setattr(signal_handler, 'num_terminated_procs', 0)
    signal_handler.num_terminated_procs += 1

signal.signal(signal.SIGCHLD, signal_handler)


def main():
    backed_up_checkpoints = []
    with open(CHECKPOINT_STATE_FILE, "r") as reader:
        for line in reader:
            backed_up_checkpoints.append(line.strip())
    print("Backed up checkpoints: {}".format(backed_up_checkpoints))
    
    current_checkpoints = os.listdir(CHECKPOINT_DIR)
    print("Current checkpoints: {}".format(current_checkpoints))
    proc_tracker = []
    cp_to_be_backuped = []
    for cp in current_checkpoints:
        if cp not in backed_up_checkpoints:
            cp_steps = int(cp[len("checkpoint-"):])
            proc_tracker.append(subprocess.Popen(["scp", os.path.join(CHECKPOINT_DIR, cp, 'pytorch_model.bin'),
                "cist-P100:/media/lvloi/backup/A100/vlsp-2022/seq2seq/tmp/pytorch_model_{}.bin".format(cp_steps)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT))
            print(proc_tracker[-1].pid)
            cp_to_be_backuped.append(cp)
    cp_to_be_backuped = sorted(cp_to_be_backuped, key=lambda x: int(x[len('checkpoint-'):]))
    with open(CHECKPOINT_STATE_FILE, "a") as writer:
        for cp in cp_to_be_backuped:
            writer.write(cp + "\n")
    
    count = 0
    while True:
        print("Waiting for data to be backed up: {}s".format(count + 1), end="\r")
        if getattr(signal_handler, 'num_terminated_procs', 0) == len(proc_tracker):
            break
        time.sleep(1.0)
        count += 1
    
    print("\nDone backup data.")


if __name__ == "__main__":
    main()
