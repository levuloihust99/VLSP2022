#!/usr/bin/python3

import os
import subprocess
import signal
import time
import argparse


def signal_handler(signalnum, stackframe):
    print("\nOne subprocess terminated.\r")
    if not hasattr(signal_handler, 'num_terminated_procs'):
        setattr(signal_handler, 'num_terminated_procs', 0)
    signal_handler.num_terminated_procs += 1

signal.signal(signal.SIGCHLD, signal_handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-state-file", required=True)
    parser.add_argument("--remote-dir", required=True)
    args = parser.parse_args()

    backed_up_checkpoints = []
    with open(args.checkpoint_state_file, "r") as reader:
        for line in reader:
            backed_up_checkpoints.append(line.strip())
    print("Backed up checkpoints: {}".format(backed_up_checkpoints))
    
    current_checkpoints = os.listdir(args.checkpoint_dir)
    print("Current checkpoints: {}".format(current_checkpoints))
    proc_tracker = []
    cp_to_be_backuped = []
    for cp in current_checkpoints:
        if cp not in backed_up_checkpoints:
            cp_steps = int(cp[len("checkpoint-"):])
            proc_tracker.append(subprocess.Popen(["scp", os.path.join(args.checkpoint_dir, cp, 'pytorch_model.bin'),
                os.path.join(args.remote_dir, "pytorch_model_{}.bin".format(cp_steps))],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT))
            print(proc_tracker[-1].pid)
            cp_to_be_backuped.append(cp)
    cp_to_be_backuped = sorted(cp_to_be_backuped, key=lambda x: int(x[len('checkpoint-'):]))
    with open(args.checkpoint_state_file, "a") as writer:
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
