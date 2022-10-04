import torch
import logging

from typing import Dict, Text
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

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
        config: TrainingConfig,
        device,
        gpu_rank: int,
        nb_gpu: int,
        done_epochs: int = 0,
        done_steps: int = 0,
        global_step: int = 0,
        number_of_updates: int = 0
    ):
        self.summarizer = summarizer
        self.optimizer = optimizer
        self.data_loader = data_loader
        self.config = config
        self.device = device
        self.gpu_rank = gpu_rank
        self.nb_gpu = nb_gpu
        self.done_epochs = done_epochs
        self.done_steps = done_steps
        self.global_step = global_step
        self.number_of_updates = number_of_updates
    
    def save(self):
        checkpoint_path = self.config.checkpoint_path
        state = {
            'done_epochs': self.done_epochs,
            'done_steps': self.done_steps,
            'global_step': self.global_step,
            'number_of_updates': self.number_of_updates,
            'model': self.summarizer.state_dict(),
            'optimizer': {k: v.state_dict() for k, v in self.optimizer.optimizers.items()},
            'scheduler': {k: v.state_dict() for k, v in self.optimizer.schedulers.items()},
            'params': self.config.to_json()
        }
        logger.info("Saving checkpoint to {}".format(checkpoint_path))
        torch.save(state, checkpoint_path)

    def train(self):
        logger.info("************************ Start training ************************")
        logger.info("Model has been trained for {} epochs and {} steps.".format(self.done_epochs, self.done_steps))
        logger.info("Global step: {} - Number of updates: {}".format(self.global_step, self.number_of_updates))
        
        rank = min(self.nb_gpu, 0)
        loss_fct = CrossEntropyLoss()
        for epoch in range(self.done_epochs, self.config.num_train_epochs):
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

                if (step + 1) % self.config.gradient_accumulate_steps == 0:
                    self.optimizer.step()
                    self.number_of_updates += 1
                
                self.global_step += 1
                self.done_steps += 1
                
                if (step + 1) % self.config.save_checkpoint_steps == 0:
                    self.save()

            self.done_steps = 0
            self.done_epochs += 1
