"""Script to process VietnameseMDS data."""

import os
import re
import json
import argparse
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="../data/KTLAB-200document-clusters/clusters")
    parser.add_argument("--output-path", required=True)
    
    args = parser.parse_args()

    clusters = os.listdir(args.input_dir)
    clusters = [c for c in clusters if c.startswith("cluster_")]
    clusters = sorted(clusters, key=lambda x: int(x[len("cluster_"):]))

    writer = open(args.output_path, "w")
    data = []
    for cluster in tqdm(clusters):
        cur_dir = os.path.join(args.input_dir, cluster)
        all_files = os.listdir(cur_dir)
        files = [f for f in all_files if f.endswith(".body.txt")]
        files = sorted(files, key=lambda x: int(x[:-len(".body.txt")]))
        docs = []
        for f in files:
            with open(os.path.join(cur_dir, f), "r") as reader:
                docs.append({'raw_text': reader.read().strip()})
        files = [f for f in all_files if f.endswith(".ref1.txt")]
        assert len(files) == 1
        f = os.path.join(cur_dir, files[0])
        with open(f, "r") as reader:
            summary = reader.read().strip()
        
        files = [f for f in all_files if f.endswith(".ref2.txt")]
        assert len(files) == 1
        f = os.path.join(cur_dir, files[0])
        with open(f, "r") as reader:
            other_summary = reader.read().strip()

        files = [f for f in all_files if f.endswith(".sum.txt")]
        assert len(files) == 1
        f = os.path.join(cur_dir, files[0])
        with open(f, "r") as reader:
            machine_summary = reader.read().strip()

        item = {'single_documents': docs, 'summary': summary, 'other_summary': other_summary, 'machine_summary': machine_summary}
        data.append(item)
        writer.write(json.dumps(item) + "\n")
    writer.close()


if __name__ == "__main__":
    main()