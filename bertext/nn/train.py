import os
import json
import hydra
import torch
import argparse
import logging

from functools import partial
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer
from torch.utils.data import Dataset, DataLoader

from .optimization import get_optimizer, get_schedule_linear
from .modeling import create_model
from .trainer import BertExtractiveTrainer
from .seeding import setup_seeding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)


def collate_fn(items, tokenizer, max_length):
    num_sents = []
    labels = []
    all_input_ids = []
    all_attn_mask = []
    max_sent_len = 0
    for item in items:
        num_sents.append(len(item))
        for subitem in item:
            labels.append(subitem['label'])
            all_input_ids.append(subitem['sentence/input_ids'])
            sent_len = len(subitem['sentence/input_ids'])
            if max_sent_len < sent_len:
                max_sent_len = sent_len
    
    max_sent_len = min(max_sent_len, max_length)
    all_input_ids_padded = []
    all_attn_mask_padded = []
    for input_ids in all_input_ids:
        input_ids = input_ids[:max_sent_len]
        input_ids[-1] = tokenizer.sep_token_id
        padding_len = max_sent_len - len(input_ids)
        attn_mask = [1] * len(input_ids) + [0] * padding_len
        all_attn_mask.append(attn_mask)
        input_ids += [tokenizer.pad_token_id] * padding_len
        all_input_ids_padded.append(input_ids)
        all_attn_mask_padded.append(attn_mask)
    
    all_input_ids_padded = torch.tensor(all_input_ids_padded)
    all_attn_mask_padded = torch.tensor(all_attn_mask_padded)

    return {
        'sentence/input_ids': all_input_ids_padded,
        'sentence/attn_mask': all_attn_mask_padded,
        'labels': torch.tensor(labels),
        'num_sents': torch.tensor(num_sents)
    }


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # setup random
    setup_seeding(cfg.seed)

    # dataset
    data = []
    with open(cfg.data_path, "r") as reader:
        for line in reader:
            data.append(json.loads(line.strip()))
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_path)
    except Exception:
        tokenizer = BertTokenizer.from_pretrained(cfg.tokenizer_path)
    tokenized_data = []
    max_seq_len = 0
    max_len_sentence = ""
    for item in tqdm(data):
        output_item = []
        for subitem in item:
            sent_token_ids = tokenizer(subitem['sentence']).input_ids
            if max_seq_len < len(sent_token_ids):
                max_seq_len = len(sent_token_ids)
                max_len_sentence = subitem['sentence']
            output_item.append({'sentence/input_ids': sent_token_ids, 'label': subitem['label']})
        tokenized_data.append(output_item)
    logger.debug("Max sentence length: {}".format(max_seq_len))
    logger.debug("Sentence with max len: {}".format(max_len_sentence))
    
    dataset = TrainDataset(tokenized_data)
    wrapped_collate_fn = partial(collate_fn, tokenizer=tokenizer, max_length=cfg.max_length)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=wrapped_collate_fn, shuffle=True)

    # model
    model = create_model(cfg)
    # checkpoint state
    try:
        files = os.listdir(cfg.output_dir)
    except FileNotFoundError:
        files = []
    files = [os.path.join(cfg.output_dir, f) for f in files]
    files = sorted(files, key=lambda x: os.path.getmtime(x), reverse=True)
    if files:
        cp_dir = files[0]
        saved_state = torch.load(os.path.join(cp_dir, 'checkpoint.pt'), map_location=lambda s, t: s)
    else:
        saved_state = None
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # optimizer
    total_updates = (len(dataset) * cfg.num_train_epochs - 1) // (cfg.batch_size * cfg.gradient_accumulate_steps) + 1
    with open_dict(cfg):
        cfg.total_updates = total_updates
    cfg.adam_betas = tuple(eval(cfg.adam_betas))
    optimizer = get_optimizer(
        model,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        adam_epsilon=cfg.adam_epsilon,
        betas=cfg.adam_betas,
    )
    scheduler = get_schedule_linear(optimizer, warmup_steps=cfg.warmup_steps, training_steps=total_updates)

    # restore checkpoint state
    trained_steps = 0
    if saved_state:
        # restore model state
        logger.info("Loading saved model state...")
        model.load_state_dict(saved_state['model_dict'])
        logger.info("Loading saved optimizer state...")
        optimizer.load_state_dict(saved_state['optimizer_dict'])
        logger.info("Loading scheduler state...")
        shift = saved_state['scheduler_dict']['last_epoch']
        scheduler = get_schedule_linear(
            optimizer,
            warmup_steps=cfg.warmup_steps,
            training_steps=total_updates,
            steps_shift=shift
        )
        trained_steps = saved_state['step']

    trainer = BertExtractiveTrainer(
        model, cfg, optimizer, scheduler, dataloader, device, {'trained_steps': trained_steps}
    )
    trainer.train()


if __name__ == '__main__':
    main()
