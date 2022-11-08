import os
import time
import logging
import json
import random
import numpy as np
import shutil

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from rouge_metric import PyRouge
from tqdm import tqdm

from model import RankingLoss
from data_utils import to_cuda
from utils import Recorder
from ..utils.dist_utils import all_gather_list

logger = logging.getLogger(__name__)

rouge = PyRouge(rouge_n=(1, 2, 4), rouge_l=True, rouge_w=True,
                    rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4)


class BRIOTrainer(object):
    def __init__(
        self,
        model,
        cfg,
        total_updates: int,
        dataloader,
        val_dataloader,
        val_gen_dataloader,
        mle_fn,
        optimizer,
        scheduler,
        scaler,
        tokenizer,
        training_state,
        recorder: Recorder,
        is_mp: bool,
        is_master: bool,
        device,
        rank: int,
        run_id: int
    ):
        self.model = model
        self.cfg = cfg
        self.total_updates = total_updates
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader
        self.val_gen_dataloader = val_gen_dataloader
        self.mle_fn = mle_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.tokenizer = tokenizer
        self.training_state = training_state
        self.recorder = recorder
        self.is_mp = is_mp
        self.is_master = is_master
        self.device = device
        self.rank = rank
        self.run_id = run_id
    
    def train(self):
        global_step = self.training_state["global_step"]
        trained_epoch = self.training_state["epoch"]
        data_step = self.training_state["data_step"]

        logger.info("*********************** Start training ***********************")
        logger.info("Num examples = {}".format(len(self.dataloader.dataset)))
        logger.info("Number of train epochs = {}".format(self.cfg.epoch))
        logger.info("Number of optimization step = {}".format(self.total_updates))
        logger.info("Number of warmup steps = {}".format(self.cfg.warmup_steps))
        logger.info("Instantaneous batch size per device = {}".format(self.cfg.batch_size))
        logger.info("Gradient accumulation steps = {}".format(self.cfg.accumulate_step))
        logger.info("Total train batch size (distributed & accumulation) = {}"
            .format(self.cfg.batch_size * self.cfg.accumulate_step * len(self.cfg.gpuid)))

        if trained_epoch > 0 or data_step > 0:
            logger.info("\tModel has been trained for {} epochs and {} data steps.".format(trained_epoch, data_step))

        for epoch in range(trained_epoch, self.cfg.epoch):
            logger.info("**************************** EPOCH {}/{} ****************************".format(epoch + 1, self.cfg.epoch))
            self.optimizer.zero_grad()
            avg_ranking_loss = 0
            avg_mle_loss = 0
            step_count = 0
            assert data_step % self.cfg.accumulate_step == 0, \
                ("data_step={}, not divisible by accumulate_step={}. "
                "This is likely due to the fact that checkpoint was saved during gradient accumulation. "
                "You should only save checkpoint after a proper update.")
            epoch_step = data_step // self.cfg.accumulate_step
            avg_loss = 0
            self.training_state['epoch'] = epoch
            t0 = time.perf_counter()

            if self.is_mp:
                self.dataloader.sampler.set_epoch(epoch)
            data_iterator = iter(self.dataloader)
            if data_step > 0:
                for _ in range(data_step):
                    next(data_iterator)

            for i, batch in enumerate(data_iterator):
                i += data_step
                self.training_state['data_step'] = i + 1
                if self.cfg.cuda:
                    to_cuda(batch, self.device)
                step_count += 1

                # forward pass
                if self.cfg.fp16:
                    with autocast():
                        output = self.model(batch["src_input_ids"], batch["candidate_ids"],
                            self.cfg.normalize, self.cfg.score_mode, self.cfg.length_penalty, adding=self.cfg.adding)
                else:
                    output = self.model(batch["src_input_ids"], batch["candidate_ids"],
                        self.cfg.normalize, self.cfg.score_mode, self.cfg.length_penalty, adding=self.cfg.adding)

                similarity, gold_similarity = output['score'], output['summary_score']
                similarity = similarity * self.cfg.scale
                gold_similarity = gold_similarity * self.cfg.scale

                ranking_loss = RankingLoss(similarity, gold_similarity, self.cfg.margin, self.cfg.gold_margin, self.cfg.gold_weight)

                probs = output["probs"]  # [bz, seq_len, word_num]
                probs = output["probs"][:, :-1]  # truncate last token

                gold = batch["candidate_ids"][:, 0, 1:]  # shift right

                mle_loss = self.mle_fn(probs.transpose(1, 2), gold)
                loss = self.cfg.rank_weight * ranking_loss + self.cfg.mle_weight * mle_loss
                loss = loss / self.cfg.accumulate_step

                avg_loss += loss.item()
                avg_mle_loss += mle_loss.item() / self.cfg.accumulate_step
                avg_ranking_loss += ranking_loss.item() / self.cfg.accumulate_step

                if self.cfg.fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if step_count == self.cfg.accumulate_step:
                    # updating
                    if self.cfg.fp16:
                        self.scaler.unscale_(self.optimizer)
                    if self.cfg.grad_norm > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_norm)
                    step_count = 0
                    epoch_step += 1
                    global_step += 1
                    self.training_state['global_step'] = global_step

                    if self.cfg.fp16:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if epoch_step % self.cfg.report_freq == 0 and step_count == 0 and self.is_master:
                    # report stats
                    logger.info("id: {}".format(self.run_id))
                    logger.info(f"similarity: {similarity[:, :10]}")

                    if not self.cfg.no_gold:
                        logger.info(f"gold similarity: {gold_similarity}")

                    self.recorder.print("epoch: %d, batch: %d, avg loss: %.6f, avg ranking loss: %.6f, avg mle loss: %.6f"
                        % (epoch + 1, epoch_step, avg_loss / self.cfg.report_freq,
                                avg_ranking_loss / self.cfg.report_freq, avg_mle_loss / self.cfg.report_freq))

                    self.recorder.print(f"learning rate: {self.scheduler.get_last_lr()[0]:.6f}")
                    logs = {"train/learning_rate": self.scheduler.get_last_lr()[0], "train/loss": avg_loss / self.cfg.report_freq,
                        "train/mle_loss": avg_mle_loss /self.cfg.report_freq, "train/ranking_loss": avg_ranking_loss / self.cfg.report_freq}
                    self.recorder.write_log(logs, global_step)
                    self.recorder.print()

                    avg_mle_loss, avg_ranking_loss, avg_loss = 0, 0, 0

                    logger.info("\x1b[38;5;3mElapsed time: {}\x1b[0m".format(time.perf_counter() - t0))
                    logger.info("\x1b[38;5;3m----------------------------------------------\x1b[0m")
                    t0 = time.perf_counter()

                del similarity, gold_similarity, loss, mle_loss, ranking_loss, output, probs

                if global_step % self.cfg.eval_interval == 0 and global_step != 0 and step_count == 0:
                    # evaluate the model as a scorer
                    scoring_metrics = self.scoring_evaluate()
                    formatted_scoring_metrics = {f"eval/scoring/{k}": v for k, v in scoring_metrics.items()}
                    self.recorder.write_log(formatted_scoring_metrics, global_step)

                    if self.is_master:
                        self.recorder.print()
                    
                    if self.training_state["best_metric"]["scoring/avg_rank"] > scoring_metrics['avg_rank']:
                        self.training_state["best_metric"]["scoring/avg_rank"] = scoring_metrics['avg_rank']
                        cp_name = "checkpoint-{}".format(global_step)
                        self.training_state["best_checkpoint"]["scoring"] = cp_name
                        self.recorder.print(
                            "best ranking metric - epoch: %d, batch: %d" % (epoch, i / self.cfg.accumulate_step))

                    if self.is_master:
                        self.recorder.print("val ranking metric: {}".format(scoring_metrics))

                    # evaluate the model as a generator
                    if self.cfg.do_generate:
                        generation_metrics = self.generation_evaluate()
                        formatted_generation_metrics = {f"eval/generation/{k}": v for k, v in generation_metrics.items()}
                        self.recorder.write_log(formatted_generation_metrics, global_step)
                        self.recorder.print()

                        if self.training_state["best_metric"]["generation/rouge-2-f"] < generation_metrics["rouge-2-f"]:
                            self.training_state["best_metric"]["generation/rouge-2-f"] = generation_metrics["rouge-2-f"]
                            cp_name = "checkpoint-{}".format(global_step)
                            self.training_state["best_checkpoint"]["generation"] = cp_name
                            self.recorder.print(
                                "best generation metric - epoch: %d, batch: %d" % (epoch, i / self.cfg.accumulate_step))
                    
                    if self.is_master:
                        self.save_checkpoint()

    def generation_evaluate(self):
        self.model.generation_mode() # switch to generation mode
        hypotheses = []
        references = []

        with torch.no_grad():
            for (i, batch) in tqdm(enumerate(self.val_gen_dataloader), total=len(self.val_gen_dataloader)):
                if self.cfg.cuda:
                    to_cuda(batch, self.device)
                samples = batch["data"]
                slines = [" ".join(x["article_untok"]) for x in samples]
                dct = self.tokenizer.batch_encode_plus(slines, max_length=self.cfg.max_doc_len, return_tensors="pt", pad_to_max_length=True, truncation=True)
                summaries = self.model.generate(
                    input_ids=dct["input_ids"].to(self.device),
                    attention_mask=dct["attention_mask"].to(self.device),
                    max_length=500,
                )
                batch_hypotheses = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
                batch_references = [[" ".join(x['abstract_untok'])] for x in samples]
                hypotheses.extend(batch_hypotheses)
                references.extend(batch_references)

        self.model.scoring_mode() # switch to scoring mode

        global_hypotheses = []
        global_references = [hasattr]
        if len(self.cfg.gpuid) > 1:
            global_hypotheses_and_references = all_gather_list([hypotheses, references])
            for hyps, refs in global_hypotheses_and_references:
                global_hypotheses.extend(hyps)
                global_references.extend(refs)
        else:
            global_hypotheses = hypotheses
            global_references = references
        
        metrics = rouge.evaluate(global_hypotheses, global_references)
        output = {}
        for metric_type, metric_value in metrics.items():
            for subtype, subvalue in metric_value.items():
                output[f"{metric_type}-{subtype}"] = subvalue

        return output

    def scoring_evaluate(self):
        all_rank_ids = []
        mle_loss = 0.0
        count = 0
        with torch.no_grad():
            for (i, batch) in tqdm(enumerate(self.val_dataloader), total=len(self.val_dataloader)):
                if self.cfg.cuda:
                    to_cuda(batch, self.device)
                output = self.model(
                    batch["src_input_ids"], batch["candidate_ids"], 
                    self.cfg.normalize, self.cfg.score_mode, self.cfg.length_penalty, adding=self.cfg.adding)

                similarity, gold_similarity = output['score'], output['summary_score']
                similarity = similarity * self.cfg.scale
                gold_similarity = gold_similarity * self.cfg.scale
                similarity = similarity.cpu().numpy() # [bz, cand]

                probs = output["probs"]  # [bz, seq_len, word_num]
                probs = output["probs"][:, :-1]  # truncate last token
                gold = batch["candidate_ids"][:, 0, 1:]  # shift right

                mle_loss += self.mle_fn(probs.transpose(1, 2), gold)
                if i % 1000 == 0:
                    logger.info(f"test similarity: {similarity[0]}")
                max_ids = similarity.argmax(1) # [bz]
                count += max_ids.shape[0]
                all_rank_ids.append(max_ids)
        
        all_rank_ids = np.concatenate(all_rank_ids, axis=0)
        avg_rank = (all_rank_ids + 1).sum() / all_rank_ids.shape[0]
        mle_loss = mle_loss / count
        return {"avg_rank": avg_rank, "mle_loss": mle_loss}

    def save_checkpoint(self):
        checkpoint_dir = os.path.join(self.recorder.dir, "checkpoints")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        cp_name = f"checkpoint-{self.training_state['global_step']}"
        checkpoint_path = os.path.join(checkpoint_dir, cp_name)

        # save model
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.model.save_pretrained(checkpoint_path)

        # save optimizer
        torch.save(self.optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))

        # save scheduler
        torch.save(self.scheduler.state_dict(), os.path.join(checkpoint_path, "scheduler.pt"))

        # save training state
        with open(os.path.join(checkpoint_path, "training_state.json"), "w") as writer:
            json.dump(self.training_state, writer, indent=4)
        
        # save RNG state
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_states["cuda"] = torch.cuda.random.get_rng_state()
        if self.is_mp:
            all_rng_states = all_gather_list([rng_states, self.rank])
            for rng_states, rank in all_rng_states:
                torch.save(rng_states, os.path.join(checkpoint_path, f"rng_state_{rank}.pth"))
        else:
            torch.save(rng_states, os.path.join(checkpoint_path, "rng_state.pth"))

        # save grad scaler state
        if self.cfg.fp16:
            torch.save(self.scaler.state_dict(), os.path.join(checkpoint_path, "scaler.pt"))
        
        # delete old checkpoint
        all_checkpoints = os.listdir(checkpoint_dir)
        all_checkpoints = [os.path.join(checkpoint_dir, cp_name) for cp_name in all_checkpoints]
        all_checkpoints = sorted(all_checkpoints, key=lambda x: os.path.getctime(x), reverse=True)

        # always keep these 2 best checkpoints
        best_generation_checkpoint = self.training_state["best_checkpoint"]["generation"]
        best_scoring_checkpoint = self.training_state["best_checkpoint"]["scoring"]

        best_generation_checkpoint = os.path.join(checkpoint_dir,
            best_generation_checkpoint)
        best_scoring_checkpoint = os.path.join(checkpoint_dir,
            best_scoring_checkpoint)
        
        kept_checkpoints = {best_generation_checkpoint, best_scoring_checkpoint}
        for cp in all_checkpoints:
            if len(kept_checkpoints) >= self.cfg.keep_checkpoint_max:
                break
            if cp not in kept_checkpoints:
                kept_checkpoints.add(cp)
        
        tobe_removed_checkpoints = [cp for cp in all_checkpoints if cp not in kept_checkpoints]
        for cp in tobe_removed_checkpoints:
            logger.info("Deleting {} since maximum kept checkpoints is 5...".format(cp))
