import argparse
import json


def preprocess_fn(tokenizer, features):
    documents = []
    summaries = []
    for docs, summary in zip(features['single_documents'], features['summary']):
        raw_texts = [doc['raw_text'] for doc in docs]
        documents.extend(raw_texts)
        summaries.extend([summary for _ in range(len(raw_texts))])
        
    documents = tokenizer(documents)
    summaries = tokenizer(summaries)

    return {'document/input_ids': documents.input_ids, 'summary/input_ids': summaries.input_ids}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True,
                        help="Path to the multi-document summarization dataset, in jsonlines format.")
    parser.add_argument("--output-path", required=True,
                        help="Path to the output flatten document-summary pair file, in jsonlines format.")
    args = parser.parse_args()

    data = []
    with open(args.input_path, "r") as reader:
        for line in reader:
            data.append(json.loads(line.strip()))

    documents = []
    summaries= []
    for item in data:
        docs = item['single_documents']
        raw_texts = [doc['raw_text'] for doc in docs]
        documents.extend(raw_texts)
        summaries.extend([item['summary'] for _ in range(len(raw_texts))])
    
    writer = open(args.output_path, 'w')
    for doc, summary in zip(documents, summaries):
        writer.write(json.dumps({'document': doc, 'summary': summary}, ensure_ascii=False) + "\n")
    writer.close()


if __name__ == "__main__":
    main()
