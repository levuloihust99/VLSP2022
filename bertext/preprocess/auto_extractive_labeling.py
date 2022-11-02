import re
import json
import string
import math
import argparse
import numpy as np
from tqdm import tqdm
from underthesea import sent_tokenize, word_tokenize


def process(item):
    punc_regex = re.compile(f"[{re.escape(string.punctuation)}]")
    # 2-grams bags for summary
    summary = punc_regex.sub("", item['summary']).lower().strip()
    summary_words = summary.split()
    summary_2_grams = [(summary_words[i], summary_words[i + 1]) for i in range(len(summary_words) - 1)]

    all_sents = []
    for doc in item['single_documents']:
        sents = sent_tokenize(doc['raw_text'])
        extended_sents = []
        for sent in sents:
            sent = re.sub(r"\n\s*", "\n", sent)
            splits = sent.split('\n')
            extended_sents.extend(splits)
        all_sents.extend(extended_sents)
    
    all_sent_2_grams = []
    for sent in all_sents:
        sent_cleaned = punc_regex.sub("", sent).lower().strip()
        words = sent_cleaned.split()
        if len(words) < 2:
            _2_grams = [('', '')]
        else:
            _2_grams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
        all_sent_2_grams.append(_2_grams)
    
    all_sent_scores = []
    for _2_grams in all_sent_2_grams:
        overlap_percentage = 0
        for _2_gram in _2_grams:
            if _2_gram in summary_2_grams:
                overlap_percentage += 1
        overlap_percentage /= len(_2_grams)
        all_sent_scores.append(overlap_percentage)
    all_sent_scores = np.array(all_sent_scores)

    sent_ids = np.argsort(-all_sent_scores)
    picked_ids = []
    n_summary_words = len(summary_words)
    words_count = 0
    for _id in sent_ids:
        sent = all_sents[_id]
        sent_cleaned = punc_regex.sub("", sent).lower().strip()
        words = sent_cleaned.split()
        words_count += len(words)
        picked_ids.append(_id)
        if words_count > n_summary_words:
            break
    picked_ids = np.array(picked_ids)
    labels = np.zeros(len(all_sents), dtype=np.int64)
    labels[picked_ids] = 1
    auto_extractive_labeled_data = []
    for sent, score, lb in zip(all_sents, all_sent_scores, labels):
        auto_extractive_labeled_data.append({'sentence': sent, 'label': int(lb), 'score': score})

    return auto_extractive_labeled_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True,
                        help="Path to the original training file provided by VLSP, in jsonlines format")
    parser.add_argument("--output-path", required=True,
                        help="Path to the output extracted file. This file contains documents whose length is reduced, ready for training abmusu.")
    args = parser.parse_args()

    data = []
    with open(args.input_path, "r") as reader:
        for line in reader:
            data.append(json.loads(line.strip()))
    
    writer = open(args.output_path, "w")
    for item in tqdm(data):
        output_item = process(item)
        writer.write(json.dumps(output_item) + "\n")
    writer.close()


if __name__ == "__main__":
    main()
