import csv
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments
from tqdm import tqdm
from torch.utils.data import DataLoader


def main():
    model_name = "VietAI/vit5-base-vietnews-summarization" # or "VietAI/vit5-large-vietnews-summarization"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    data_path = "/home/lvloi/projects/vlsp-2022/data/vlsp-2022/pretokenized/vlsp_2022_absumu_train_data_pretokenized.jsonl"
    