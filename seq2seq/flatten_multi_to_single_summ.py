import csv
import json
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


def main():
    model_name = "VietAI/vit5-base-vietnews-summarization" # or "VietAI/vit5-large-vietnews-summarization"
    tokenizer = AutoTokenizer.from_pretrained(model_name)  

    data_path = "/media/lvloi/projects/vlsp-2022/data/vlsp-2022/jsonl/vlsp_2022_abmusu_train_data_new.jsonl"
    data = []
    with open(data_path, "r") as reader:
        for line in reader:
            data.append(json.loads(line.strip()))

    documents = []
    summaries= []
    for item in data:
        docs = item['single_documents']
        raw_texts = [doc['raw_text'] for doc in docs]
        documents.extend(raw_texts)
        summaries.extend([item['summary'] for _ in range(len(raw_texts))])
    
    output_path = "/media/lvloi/projects/vlsp-2022/data/vlsp-2022/jsonl/vlsp_2022_abmusu_train_data_flattened.jsonl"
    writer = open(output_path, 'w')
    for doc, summary in zip(documents, summaries):
        writer.write(json.dumps({'document': doc, 'summary': summary}, ensure_ascii=False) + "\n")
    writer.close()


if __name__ == "__main__":
    main()
