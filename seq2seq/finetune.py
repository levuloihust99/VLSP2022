import csv
import json
import hydra
import torch

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments
from tqdm import tqdm
from torch.utils.data import DataLoader
from omegaconf import DictConfig


def process_fn(feature):
    input_ids = torch.tensor(feature['document/input_ids'])
    labels = torch.tensor(feature['summary/input_ids'])

    return {
        'input_ids': input_ids,
        'labels': labels
    }


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)

    data = []
    with open(cfg.data_path, "r") as reader:
        for line in reader:
            data.append(json.loads(line.strip()))
    dataset = Dataset.from_list(data)
    tokenized_dataset = dataset.map(process_fn)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")
    
    training_args = Seq2SeqTrainingArguments(
        "tmp/",
        do_train=True,
        do_eval=False,
        num_train_epochs=3,
        learning_rate=1e-5,
        warmup_ratio=0.05,
        weight_decay=0.01,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        logging_dir="./logs",
        group_by_length=False,
        save_strategy='epoch',
        save_total_limit=3,
        fp16=True,
        gradient_accumulation_steps=5
    )

    trainer = trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    main()
