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

    data_path = "/home/lvloi/projects/vlsp-2022/data/vlsp-2022/jsonl/vlsp_2022_abmusu_train_data_new.jsonl"
    data = []
    with open(data_path, "r") as reader:
        for line in reader:
            data.append(json.loads(line.strip()))

    dataset = Dataset.from_list(data)
    dataset = dataset.map(lambda features: preprocess_fn(tokenizer, features), batched=True,
        num_proc=10, batch_size=10, remove_columns=['single_documents', 'summary', 'category'])
    
    writer = open("/home/lvloi/projects/vlsp-2022/data/vlsp-2022/pretokenized/vlsp_2022_absumu_train_data_pretokenized.jsonl", "w")
    for idx in range(len(dataset)):
        writer.write(json.dumps(dataset[idx]) + "\n")


if __name__ == "__main__":
    main()
