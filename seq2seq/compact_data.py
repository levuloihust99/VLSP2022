import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-file", required=True,
                        help="Path to the textline candidates file.")
    parser.add_argument("--src-path", required=True,
                        help="Path to the source data file.")
    parser.add_argument("--output-path", required=True,
                        help="Path to output file.")
    args = parser.parse_args()

    src_data = []
    with open(args.src_path, "r") as reader:
        for line in reader:
            src_data.append(line.strip())
    
    candidates = []
    with open(args.candidate_file, "r") as reader:
        for line in reader:
            candidates.append(line.strip())
    
    out_data = []
    for src, cand in zip(src_data, candidates):
        out_data.append({"src": src, "candidate": cand})

    with open(args.output_path, "w") as writer:
        json.dump(out_data, writer, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
