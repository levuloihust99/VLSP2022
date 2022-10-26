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
    tokenized_dataset = dataset.map(process_fn, remove_columns=['document/input_ids', 'summary/input_ids'], batched=False)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")
    
    training_args = Seq2SeqTrainingArguments(
        "tmp/",
        do_train=cfg.do_train,
        do_eval=cfg.do_eval,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        logging_dir=cfg.logging_dir,
        group_by_length=cfg.group_by_length,
        save_strategy=cfg.save_strategy,
        save_total_limit=cfg.save_total_limit,
        fp16=cfg.fp16,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    main()
