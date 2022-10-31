import argparse
import json
from tqdm import tqdm
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True,
                        help="Path to abmusu test file, in jsonlines format.")
    parser.add_argument("--output-path", required=True,
                        help="Path to output textline file.")
    parser.add_argument("--tokenizer-path", default="VietAI/vit5-base-vietnews-summarization")
    args = parser.parse_args()

    data = []
    with open(args.input_path, "r") as reader:
        for line in reader:
            data.append(json.loads(line.strip()))
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    with open(args.output_path, "w") as writer:
        for item in tqdm(data):
            raw_text = item['single_documents'][0]['raw_text']
            token_ids = tokenizer(raw_text).input_ids
            decoded_text = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False, skip_special_tokens=True)
            writer.write(decoded_text + "\n")


if __name__ == "__main__":
    main()
