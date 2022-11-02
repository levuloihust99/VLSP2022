import re
import json
import string
import math
import argparse
import numpy as np
from tqdm import tqdm
from nltk.tokenize import sent_tokenize, word_tokenize


def process(item):
    punc_regex = re.compile(f"[{re.escape(string.punctuation)}]")
    # 2-grams bags for summary
    summary = punc_regex.sub("", item['summary']).lower().strip()
    words = summary.split()
    summary_2_grams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
    
    doc_2_grams = []
    doc_sents = []
    for doc in item['single_documents']:
        sents = sent_tokenize(doc['raw_text'])
        doc_sents.append(sents)
        sent_2_grams = []
        for sent in sents:
            text = punc_regex.sub("", sent).lower().strip()
            words = text.split()
            if len(words) < 2:
                _2_grams = [('', '')]
            else:
                _2_grams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
            sent_2_grams.append(_2_grams)
        doc_2_grams.append(sent_2_grams)
    
    extracted_docs = []
    for idx, sent_2_grams in enumerate(doc_2_grams):
        sent_scores = []
        for _2_grams in sent_2_grams:
            overlap_percentage = 0
            for _2_gram in _2_grams:
                if _2_gram in summary_2_grams:
                    overlap_percentage += 1
            overlap_percentage /= len(_2_grams)
            sent_scores.append(overlap_percentage)
        sent_scores = np.array(sent_scores)
        sorted_ids = np.argsort(-sent_scores)
        num_sent = len(sorted_ids)
        num_picked_sent = math.ceil(num_sent // 2)
        picked_ids = sorted(sorted_ids[:num_picked_sent].tolist())
        picked_sents = [doc_sents[idx][_id] for _id in picked_ids]
        extracted_docs.append(" ".join(picked_sents))
    
    return {'documents': extracted_docs, 'summary': item['summary']}


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
