import os
import time
import torch
import logging

from tqdm import tqdm
from typing import Dict, Text, Optional, Any
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from ..utils.distributed import all_gather_list
from ..modeling.configuration import TrainingConfig
from .summarizer import AbsSummarizer
from ..modeling.optimization import AbsSummarizerOptimizer

logger = logging.getLogger(__name__)


class TrainerHelper(object):
    @staticmethod
    def chunk(batch: Dict[Text, torch.Tensor], n):
        if n < 2:
            return {k: v.unsqueeze(0) for k, v in batch.items()}
        batch_size = next(iter(batch.values())).size(0)
        per_gpu_batch_size = batch_size // n
        remainder = batch_size % n
        scatters = [1] * remainder + [0] * (n - remainder)
        per_gpu_batch_sizes = [per_gpu_batch_size] * n
        per_gpu_batch_sizes = [per_gpu_batch_size + scatter for 
                                per_gpu_batch_size, scatter in zip(per_gpu_batch_sizes, scatters)]
        splitted_batch = {
            k: torch.split(v, per_gpu_batch_sizes)
            for k, v in batch.items()
        }
        return splitted_batch


class SummarizerTrainer(object):
    def __init__(
        self,
        summarizer: AbsSummarizer,
        optimizer: AbsSummarizerOptimizer,
        data_loader: DataLoader,
        dev_dataloader: Optional[DataLoader],
        config: TrainingConfig,
        device,
        gpu_rank: int,
        nb_gpu: int,
        done_epochs: int = 0,
        done_steps: int = 0,
        global_step: int = 0,
        number_of_updates: int = 0,
        ckpt_counter: int = 0,
        best_checkpoint_name: Text = None,
        best_checkpoint_val_acc: float = 0.0
    ):
        self.summarizer = summarizer
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.dev_dataloader = dev_dataloader
        self.config = config
        self.device = device
        self.gpu_rank = gpu_rank
        self.nb_gpu = nb_gpu
        self.done_epochs = done_epochs
        self.done_steps = done_steps
        self.global_step = global_step
        self.number_of_updates = number_of_updates
        self.ckpt_counter = ckpt_counter
        self.best_checkpoint_name = best_checkpoint_name
        self.best_checkpoint_val_acc = best_checkpoint_val_acc
    
    def save(self, cp_name):
        model = self.summarizer.module if hasattr(self.summarizer, 'module') else self.summarizer
        state = {
            'done_epochs': self.done_epochs,
            'done_steps': self.done_steps,
            'global_step': self.global_step,
            'number_of_updates': self.number_of_updates,
            'ckpt_counter': self.ckpt_counter,
            'model': model.state_dict(),
            'optimizer': {k: v.state_dict() for k, v in self.optimizer.optimizers.items()},
            'scheduler': {k: v.state_dict() for k, v in self.optimizer.schedulers.items()},
            'params': self.config.to_json(),
            'best_checkpoint': {
                'name': self.best_checkpoint_name,
                'val_accuracy': self.best_checkpoint_val_acc
            }
        }
        checkpoint_dir = os.path.dirname(self.config.checkpoint_path)
        saved_checkpoint_path = os.path.join(checkpoint_dir, cp_name)
        logger.info("Saving checkpoint to {}".format(saved_checkpoint_path))
        all_checkpoints = os.listdir(checkpoint_dir)
        all_checkpoints = [os.path.join(checkpoint_dir, f) for f in all_checkpoints]
        best_cp_path = os.path.join(checkpoint_dir, self.best_checkpoint_name)
        all_checkpoints = sorted(all_checkpoints, key=lambda x: os.path.getctime(x), reverse=True)
        kept_checkpoints = all_checkpoints[:self.config.keep_checkpoint_max] + [best_cp_path]
        
        for cp in all_checkpoints:
            if cp not in kept_checkpoints:
                os.remove(cp)

        torch.save(state, saved_checkpoint_path)

    def train(self):
        logger.info("************************ Start training ************************")
        logger.info("Model has been trained for {} epochs and {} steps.".format(self.done_epochs, self.done_steps))
        logger.info("Global step: {} - Number of updates: {}".format(self.global_step, self.number_of_updates))
        
        rank = max(self.gpu_rank, 0)
        loss_fct = CrossEntropyLoss()
        total_steps = len(self.data_loader) * self.config.num_train_epochs
        logger.info("Total training steps: {}".format(total_steps))
        per_epoch_steps = len(self.data_loader)
        mark_time = time.perf_counter()
        for epoch in range(self.done_epochs, self.config.num_train_epochs):
            logger.info("--------------------- EPOCH {}/{} ---------------------".format(epoch + 1, self.config.num_train_epochs))
            self.data_loader.sampler.seed(epoch + self.config.seed)
            for step, batch in enumerate(self.data_loader):
                if step < self.done_steps:
                    continue
                batch = TrainerHelper.chunk(batch, self.nb_gpu)
                encoder_input_ids = batch['encoder_input_ids'][rank].to(self.device)
                encoder_attention_mask = batch['encoder_attention_mask'][rank].to(self.device)
                encoder_token_type_ids = batch['encoder_token_type_ids'][rank].to(self.device)
                decoder_input_ids = batch['decoder_input_ids'][rank].to(self.device)
                decoder_attention_mask = batch['decoder_attention_mask'][rank].to(self.device)
                labels = batch['labels'][rank].to(self.device)

                logits = self.summarizer(
                    input_ids=encoder_input_ids,
                    attention_mask=encoder_attention_mask,
                    token_type_ids=encoder_token_type_ids,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask
                ) # [bsz, seq_len, vocab_size]
                _, _, vocab_size = logits.size()

                active_mask = decoder_attention_mask.view(-1) == 1
                active_logits = logits.view(-1, vocab_size)[active_mask]
                active_labels = labels.view(-1)[active_mask]

                loss = loss_fct(active_logits, active_labels)
                loss = loss / self.config.gradient_accumulate_steps
                loss.backward()

                self.global_step += 1
                self.done_steps += 1

                if (step + 1) % self.config.logging_steps == 0:
                    log_string = "\n\n\tStep {}/{}".format(step + 1, per_epoch_steps)
                    log_string += "\n\tTime elapsed: {}s".format(time.perf_counter() - mark_time)
                    mark_time = time.perf_counter()
                    log_string += "\n\tLoss: {}\n".format(loss.item() * self.config.gradient_accumulate_steps)
                    logger.info(log_string)

                if self.global_step % self.config.gradient_accumulate_steps == 0:
                    self.optimizer.step()
                    self.number_of_updates += 1
                
                if self.global_step % self.config.save_checkpoint_steps == 0:
                    accuracy = self.validate()
                    cp_name = os.path.basename(self.config.checkpoint_path) + f"_{self.ckpt_counter}.pt"
                    self.ckpt_counter += 1
                    if accuracy > self.best_checkpoint_val_acc:
                        self.best_checkpoint_val_acc = accuracy
                        self.best_checkpoint_name = cp_name
                    self.save(cp_name)
                
            self.done_steps = 0
            self.done_epochs += 1

    def validate(self):
        logger.info("************************ Running validation ************************")
        start_time = time.perf_counter()
        rank = max(self.gpu_rank, 0)
        total_n_tokens = 0
        total_n_correct = 0
        with torch.no_grad():
            for batch in tqdm(self.dev_dataloader):
                origin_batch = batch # DEBUG
                batch = TrainerHelper.chunk(batch, self.nb_gpu)
                encoder_input_ids = batch['encoder_input_ids'][rank].to(self.device)
                encoder_attention_mask = batch['encoder_attention_mask'][rank].to(self.device)
                encoder_token_type_ids = batch['encoder_token_type_ids'][rank].to(self.device)
                decoder_input_ids = batch['decoder_input_ids'][rank].to(self.device)
                decoder_attention_mask = batch['decoder_attention_mask'][rank].to(self.device)
                labels = batch['labels'][rank].to(self.device)

                logits = self.summarizer(
                    input_ids=encoder_input_ids,
                    attention_mask=encoder_attention_mask,
                    token_type_ids=encoder_token_type_ids,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask
                ) # [bsz, seq_len, vocab_size]
                predictions = torch.argmax(logits, dim=-1)

                num_tokens = decoder_attention_mask.sum().item()
                active_mask = decoder_attention_mask.view(-1) == 1
                active_predictions = predictions.view(-1)[active_mask] # [bsz * seq_len][active_mask]
                active_labels = labels.view(-1)[active_mask] # [bsz * seq_len][active_mask]

                n_correct = (active_predictions == active_labels).sum().item()
                if self.nb_gpu > 1:
                    gathered = all_gather_list((num_tokens, n_correct))
                    for n_toks, n_cor in gathered:
                        total_n_tokens += n_toks
                        total_n_correct += n_cor
                else:
                    total_n_tokens += num_tokens
                    total_n_correct += n_correct

        logger.info("Validation took: {}s".format(time.perf_counter() - start_time))
        validation_result = (
            "Total tokens = {} :: Total correct = {} :: Accuracy = {}"
        ).format(total_n_tokens, total_n_correct, total_n_correct / total_n_tokens)
        logger.info("****************** Validation results ******************\n{}".format(validation_result))
        
        return total_n_correct / total_n_tokens # return accuracy
