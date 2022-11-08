import os
import time
import json
import hydra
import math
import argparse
import logging
import random
import numpy as np

from tqdm import tqdm
from omegaconf import DictConfig
from functools import partial
from nltk import sent_tokenize, word_tokenize

import torch
import torch.nn as nn
import torch.optim as optim

from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp

from transformers import BartTokenizer, PegasusTokenizer, T5Tokenizer

from own.utils.seeding import seed_everything
from own.nn.optimization import get_optimizer, get_schedule_linear
from own.nn.trainer import BRIOTrainer

from utils import Recorder
from data_utils import to_cuda, collate_mp_brio, BrioDataset
from config import base_setting, cnndm_setting, xsum_setting, abmusu_settings
from label_smoothing_loss import label_smoothing_loss
from model import RankingLoss, BRIO

logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_fast").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


def run(rank, cfg):
    # setup random
    seed_everything(cfg.seed)

    gpuid = cfg.gpuid[rank]
    if cfg.cuda and torch.cuda.is_available():
        torch.cuda.set_device(f"cuda:{gpuid}")
    is_master = rank == 0
    is_mp = len(cfg.gpuid) > 1
    world_size = len(cfg.gpuid)
    if is_master:
        id = len(os.listdir("./cache"))
        recorder = Recorder(id, cfg.log)

    # build dataloader
    assert not (cfg.is_pegasus and cfg.is_t5), "is_pegasus and is_t5 cannot be True at the same time."
    if cfg.is_pegasus:
        tokenizer = PegasusTokenizer.from_pretrained(cfg.model_type)
    elif cfg.is_t5:
        tokenizer = T5Tokenizer.from_pretrained(cfg.model_type)
    else:
        tokenizer = BartTokenizer.from_pretrained(cfg.model_type)

    collate_fn = partial(collate_mp_brio, pad_token_id=tokenizer.pad_token_id, is_test=False)
    collate_fn_val = partial(collate_mp_brio, pad_token_id=tokenizer.pad_token_id, is_test=True)

    train_set = BrioDataset(
        f"./{cfg.dataset}/{cfg.datatype}/{cfg.train_split}",
        cfg.model_type, max_summ_len=cfg.max_summ_len, max_candidates=cfg.max_candidates,
        max_doc_len=cfg.max_doc_len, is_pegasus=cfg.is_pegasus, is_t5=cfg.is_t5)
    val_set = BrioDataset(
        f"./{cfg.dataset}/{cfg.datatype}/{cfg.val_split}",
        cfg.model_type, is_test=True, max_summ_len=512, is_sorted=False,
        max_candidates=cfg.max_candidates, max_doc_len=cfg.max_doc_len, is_pegasus=cfg.is_pegasus, is_t5=cfg.is_t5)

    if is_mp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
    	                        train_set, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn, sampler=train_sampler)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
    	                        val_set, num_replicas=world_size, rank=rank)
        val_dataloader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn_val, sampler=val_sampler)
        val_gen_dataloader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn_val, sampler=val_sampler)
    else:
        # dataloader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
        dataloader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn) # debug
        val_dataloader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn_val)
        val_gen_dataloader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn_val)

    # build models
    model_path = cfg.pretrained if cfg.pretrained is not None else cfg.model_type
    model = BRIO(model_path, tokenizer.pad_token_id, is_pegasus=cfg.is_pegasus, is_t5=cfg.is_t5)

    if cfg.cuda:
        if is_mp:
            # Using DDP
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            model = nn.parallel.DistributedDataParallel(model.to(gpuid), [gpuid], find_unused_parameters=False)
        else:
            model = model.cuda()
    model.train()

    optimizer = get_optimizer(model, learning_rate=cfg.optimizer.learning_rate,
        adam_eps=cfg.optimizer.adam_eps, weight_decay=cfg.optimizer.weight_decay)

    num_replicas = 1
    if len(cfg.gpuid) > 1:
        num_replicas = len(cfg.gpuid)
    num_samples_per_replica = math.ceil(len(train_set) / num_replicas)
    updates_per_epoch = math.floor(num_samples_per_replica / (cfg.batch_size * cfg.accumulate_step))
    total_updates = updates_per_epoch * cfg.epoch
    scheduler = get_schedule_linear(optimizer, cfg.warmup_steps, total_updates)
    if is_master:
        recorder.write_config(cfg, [model], __file__)
    scaler = GradScaler() if cfg.fp16 else None
    
    # TODO: restore checkpoint
    training_state = {
        "best_metric": {
            "scoring/avg_rank": 1000,
            "generation/rouge-2-f": 0.0
        },
        "best_checkpoint": {
            "scoring": None,
            "generation": None
        },
        "global_step": 0,
        "data_step": 0,
        "epoch": 0
    }
    if cfg.resume_from_checkpoint:
        logger.info('Loading model weights ...')
        model_dict = torch.load(os.path.join(cfg.resume_from_checkpoint, "pytorch_model.bin"), map_location=lambda s, t: s)
        model.model.load_state_dict(model_dict)

        logger.info('Loading saved optimizer state ...')
        optimizer_dict = torch.load(os.path.join(cfg.resume_from_checkpoint, "optimizer.pt"), map_location=lambda s, t: s)
        optimizer.load_state_dict(optimizer_dict)

        logger.info("Loading scheduler state ...")
        scheduler_dict = torch.load(os.path.join(cfg.resume_from_checkpoint, "scheduler.pt"), map_location=lambda s, t: s)
        shift = int(scheduler_dict["last_epoch"])
        logger.info("Steps shift %d", shift)
        scheduler = get_schedule_linear(
            optimizer,
            cfg.warmup_steps,
            total_updates,
            steps_shift=shift
        )

        logger.info("Loading RNG state ...")
        if is_mp:
            rng_states = torch.load(os.path.join(cfg.resume_from_checkpoint, f"rng_state_{rank}.pth"))
        else:
            rng_states = torch.load(os.path.join(cfg.resume_from_checkpoint, "rng_state.pth"))
        random.setstate(rng_states["python"])
        np.random.set_state(rng_states["numpy"])
        torch.random.set_rng_state(rng_states["cpu"])
        if torch.cuda.is_available():
            torch.cuda.random.set_rng_state(rng_states["cuda"])
        
        logger.info("Loading training state ...")
        with open(os.path.join(cfg.resume_from_checkpoint, "training_state.json"), "r") as reader:
            training_state = json.load(reader)

        if cfg.fp16:
            logger.info("Loading GradScaler state ...")
            scaler_dict = torch.load(os.path.join(cfg.resume_from_checkpoint, "scaler.pt"))
            scaler.load_state_dict(scaler_dict)

    # set the model to scoring mode
    if is_mp:
        model.module.scoring_mode()
    else:
        model.scoring_mode()

    if cfg.smooth > 0:
        mle_fn = label_smoothing_loss(ignore_index=tokenizer.pad_token_id, epsilon=cfg.smooth)
    else:
        mle_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # instantiate trainer
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    trainer = BRIOTrainer(
        model, cfg, total_updates, dataloader,
        val_dataloader, val_gen_dataloader, mle_fn,
        optimizer, scheduler, scaler, tokenizer, training_state,
        recorder,is_mp, is_master, device, rank, id
    )

    trainer.train()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # set env
    if len(cfg.gpuid) > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = f'{cfg.port}'
        mp.spawn(run, cfg=(cfg,), nprocs=len(cfg.gpuid), join=True)
    else:
        run(0, cfg)

if __name__ ==  "__main__":
    main()
