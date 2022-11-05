import csv
import json
import hydra
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
    Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments, T5Config
)
from tqdm import tqdm
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import evaluate
import numpy as np

metric = evaluate.load('rouge')


def process_fn(feature):
    input_ids = torch.tensor(feature['document/input_ids'])
    labels = torch.tensor(feature['summary/input_ids'])

    return {
        'input_ids': input_ids,
        'labels': labels
    }


def get_compute_metrics(tokenizer):
    def compute_metrics(eval_preds):
        metric = evaluate.load("rouge")
        generated_tokens, labels = eval_preds
        with tokenizer.as_target_tokenizer():
            generated_texts = [
                tokenizer.decode(tokens, clean_up_tokenization_spaces=False, skip_special_tokens=True)
                for tokens in generated_tokens
            ]
            references = [
                tokenizer.decode(tokens, clean_up_tokenization_spaces=False, skip_special_tokens=True)
                for tokens in labels
            ]
        return metric.compute(predictions=generated_texts, references=references)
    return compute_metrics


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # print config
    for k, v in cfg.items():
        print("{}--> {}".format(k + " " * (40 - len(k)), v))

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    config = T5Config.from_pretrained(cfg.model_name, gradient_checkpointing=cfg.gradient_checkpointing)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name, config=config)

    train_data = []
    with open(cfg.train_data_path, "r") as reader:
        for line in reader:
            train_data.append(json.loads(line.strip()))
    train_dataset = Dataset.from_list(train_data)
    tokenized_train_dataset = train_dataset.map(process_fn, remove_columns=['document/input_ids', 'summary/input_ids'], batched=False)

    if cfg.do_eval:
        dev_data = []
        with open(cfg.dev_data_path, "r") as reader:
            for line in reader:
                dev_data.append(json.loads(line.strip()))
        dev_dataset = Dataset.from_list(dev_data)
        tokenized_dev_dataset = dev_dataset.map(process_fn, remove_columns=['document/input_ids', 'summary/input_ids'], batched=False)
    else:
        tokenized_dev_dataset = None

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
        evaluation_strategy=cfg.save_strategy,
        save_total_limit=cfg.save_total_limit,
        fp16=cfg.fp16,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_steps=cfg.logging_steps,
        logging_first_step=cfg.logging_first_step,
        max_grad_norm=cfg.max_grad_norm,
        label_smoothing_factor=cfg.label_smoothing_factor,
        report_to=['tensorboard'],
        load_best_model_at_end=True,
        metric_for_best_model="eval_rouge2",
        predict_with_generate=True,
        generation_max_length=1024
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_dev_dataset,
        data_collator=data_collator,
        compute_metrics=get_compute_metrics(tokenizer)
    )

    trainer.train()


if __name__ == "__main__":
    main()
