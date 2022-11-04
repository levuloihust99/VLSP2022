import json
import re
import string
import argparse
import collections
from tqdm import tqdm

punc_regex = re.compile(f"[{re.escape(string.punctuation)}]")


def normalize(text):
    text = punc_regex.sub(" ", text).lower()
    text = re.sub("\s+", " ", text)
    return text


def cal_score(cand, ref):
    norm_cand = normalize(cand)
    norm_ref = normalize(ref)

    cand_words = norm_cand.split()
    cand_bigrams = collections.Counter()
    for idx in range(len(cand_words) - 1):
        cand_bigrams[tuple(cand_words[idx : idx + 2])] += 1

    ref_words = norm_ref.split()
    ref_bigrams = collections.Counter()
    for idx in range(len(ref_words) - 1):
        ref_bigrams[tuple(ref_words[idx : idx + 2])] += 1
    
    matched_bigrams = cand_bigrams & ref_bigrams
    num_matched_bigrams = sum(matched_bigrams.values())
    num_cand_bigrams = sum(cand_bigrams.values())
    num_ref_bigrams = sum(ref_bigrams.values())

    if num_matched_bigrams == 0:
        return 0.0

    p = num_matched_bigrams / num_cand_bigrams
    r = num_matched_bigrams / num_ref_bigrams
    f = (2 * p * r) / (p + r)
    return f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True,
                        help="Path to the contrastive data, in jsonlines format. Each line "
                        "is a dumped json with following fields: 'single_documents', 'summary', 'candidates'.")
    parser.add_argument("--output-path", required=True,
                        help="Path to the output contrastive data, in jsonlines format. "
                        "Output data has the same fields as input data but additional info about candidate score.")
    args = parser.parse_args()

    input_data = []
    with open(args.input_path, "r") as reader:
        for line in reader:
            input_data.append(json.loads(line.strip()))
    
    output_data = []
    for item in tqdm(input_data):
        summary = item['summary']
        candidates = item['candidates']
        scored_candidates = []
        for cand in candidates:
            score = cal_score(cand, summary)
            scored_candidates.append({'text': cand, 'score': score})
        output_item = {**item, 'candidates': scored_candidates}
        output_data.append(output_item)

    with open(args.output_path, "w") as writer:
        for item in output_data:
            writer.write(json.dumps(item) + '\n')


if __name__ == "__main__":
    main()
