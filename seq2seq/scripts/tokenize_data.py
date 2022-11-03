import csv
import json
import argparse
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments
from tqdm import tqdm
from torch.utils.data import DataLoader


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


def preprocess_multi_doc(tokenizer, features, model_type='base'):
    documents = []
    summaries = []

    for docs, summary in zip(features['single_documents'], features['summary']):
        raw_texts = [doc['raw_text'] for doc in docs]
        concat_text = " ".join(raw_texts)
        if model_type == 'large':
            concat_text = "vietnews: " + concat_text
        documents.append(concat_text)
        summaries.append(summary)
    
    documents = tokenizer(documents, max_length=4096, truncation=True)
    summaries = tokenizer(summaries, max_length=4096, truncation=True)

    return {'document/input_ids': documents.input_ids, 'summary/input_ids': summaries.input_ids}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="VietAI/vit5-base-vietnews-summarization")
    parser.add_argument("--tokenizer-path", default="VietAI/vit5-base-vietnews-summarization")
    parser.add_argument("--model-type", choices=['base', 'large'], default='base')
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)  

    data = []
    with open(args.data_path, "r") as reader:
        for line in reader:
            data.append(json.loads(line.strip()))

    dataset = Dataset.from_list(data)
    dataset = dataset.map(lambda features: preprocess_multi_doc(tokenizer, features, args.model_type), batched=True,
        num_proc=10, batch_size=10, remove_columns=['single_documents', 'summary'])
    
    writer = open(args.output_path, "w")
    for idx in range(len(dataset)):
        item = {'document/input_ids': dataset[idx]['document/input_ids'], 'summary/input_ids': dataset[idx]['summary/input_ids']}
        writer.write(json.dumps(item) + "\n")
    writer.close()


if __name__ == "__main__":
    main()
