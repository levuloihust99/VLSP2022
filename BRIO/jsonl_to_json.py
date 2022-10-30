import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-file", required=True)
    parser.add_argument("--json-file", required=True)
    args = parser.parse_args()

    data = []
    with open(args.jsonl_file, "r") as reader:
        for line in reader:
            data.append(json.loads(line.strip()))
    
    with open(args.json_file, "w") as writer:
        json.dump(data, writer, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
