import os
import json
import argparse

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True,
                        help="Path to the input data, in jsonlines format, contain all data samples. "
                        "Data contained in this file should have candidates and candidate scores.")
    parser.add_argument("--output-dir", required=True,
                        help="Directory that will contain the data files. Structure of this directory is specified " 
                        "in BRIO.")
    parser.add_argument("--dataset-type", default="diverse",
                        help="This parameter is from BRIO.")
    parser.add_argument("--split", required=True, choices=['train', 'val', 'test'],
                        help="One of 'train', 'val' or 'test'.")
    args = parser.parse_args()

    input_data = []
    with open(args.input_path, "r") as reader:
        for line in reader:
            input_data.append(json.loads(line.strip()))
    
    output_dir = os.path.join(args.output_dir, args.dataset_type, args.split)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for idx, item in tqdm(enumerate(input_data), total=len(input_data)):
        output_item = {
            'article': [doc['raw_text'] for doc in item['single_documents']],
            'abstract': [item['summary']],
            'candidates': [
                [[cand['text']], cand['score']]
                for cand in item['candidates']
            ]
        }
        output_item = {
            **output_item,
            'article_untok': output_item['article'],
            'abstract_untok': output_item['abstract'],
            'candidates_untok': output_item['candidates']
        }

        with open(os.path.join(output_dir, "{}.json".format(idx)), "w") as writer:
            json.dump(output_item, writer, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()