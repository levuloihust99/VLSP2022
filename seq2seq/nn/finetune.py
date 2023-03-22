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
from rouge_metric import PyRouge


def process_fn(feature):
    input_ids = torch.tensor(feature['document/input_ids'])
    labels = torch.tensor(feature['summary/input_ids'])

    return {
        'input_ids': input_ids,
        'labels': labels
    }


def get_compute_metrics(tokenizer):
    def compute_metrics(eval_preds):
        generated_tokens, labels = eval_preds
        generated_tokens[generated_tokens == -100] = tokenizer.pad_token_id
        labels[labels == -100] = tokenizer.pad_token_id

        with tokenizer.as_target_tokenizer():
            generated_texts = [
                tokenizer.decode(tokens, clean_up_tokenization_spaces=False, skip_special_tokens=True)
                for tokens in generated_tokens
            ]
            try:
                references = [
                    [tokenizer.decode(tokens, clean_up_tokenization_spaces=False, skip_special_tokens=True)]
                    for tokens in labels
                ]
            except Exception as e:
                import pickle
                import traceback

                with open("tensor_state.pkl", "wb") as writer:
                    pickle.dump({'generated_tokens': generated_tokens, 'labels': labels}, writer)
                print(traceback.format_exc())
                raise KeyboardInterrupt

        rouge = PyRouge(rouge_n=(1, 2), rouge_l=True, rouge_w=True,
                    rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4)
        metrics = rouge.evaluate(generated_texts, references)
        output = {}
        for metric_type, metric_value in metrics.items():
            if metric_type in {'rouge-1', 'rouge-2', 'rouge-l'}:
                for metric_subtype, metric_subvalue in metric_value.items():
                    output["{}-{}".format(metric_type, metric_subtype)] = metric_subvalue
        return output

    return compute_metrics


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # print config
    for k, v in cfg.items():
        print("{}--> {}".format(k + " " * (40 - len(k)), v))

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
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
        cfg.output_dir,
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
        # save_strategy="steps", # for DEBUG
        # save_steps=1, # for DEBUG
        # evaluation_strategy="steps", # for DEBUG
        # eval_steps=1, # for DEBUG
        save_total_limit=cfg.save_total_limit,
        fp16=cfg.fp16,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        logging_steps=cfg.logging_steps,
        logging_first_step=cfg.logging_first_step,
        max_grad_norm=cfg.max_grad_norm,
        label_smoothing_factor=cfg.label_smoothing_factor,
        report_to=['tensorboard'],
        load_best_model_at_end=True,
        metric_for_best_model="rouge-2-f",
        predict_with_generate=True,
        generation_max_length=cfg.generation_max_length
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_dev_dataset,
        data_collator=data_collator,
        compute_metrics=get_compute_metrics(tokenizer)
    )

    trainer.train(cfg.resume_from_checkpoint)


if __name__ == "__main__":
    main()
