import torch
from transformers import AdamW, get_linear_schedule_with_warmup

from typing import Dict, Text, Any


def create_optimizers_and_schedulers(
    model,
    total_steps: int,
    weight_decay: float = 0.0,
    learning_rate: float = 5e-5,
    adam_epsilon: float = 1e-8,
    betas: tuple = (0.9, 0.999),
    num_warmup_steps: int = 1000
):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon, betas=betas)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=total_steps)
    return optimizer, scheduler


class AbsSummarizerOptimizer(object):
    def __init__(
        self,
        optimizers: Dict[Text, torch.optim.Optimizer],
        schedulers: Dict[Text, torch.optim.lr_scheduler._LRScheduler],
        max_grad_norm: float = 0.0
    ):
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.max_grad_norm = max_grad_norm

    def step(self):
        for opt in self.optimizers.values():
            if self.max_grad_norm:
                params = []
                for param_group in opt.param_groups:
                    params.extend(param_group['params'])
                torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
            opt.step()
            opt.zero_grad()
        for sche in self.schedulers.values():
            sche.step()

    def restore(self, optimizer_state: Dict[Text, Any], scheduler_state: Dict[Text, Any]):
        # key matching check
        optimizer_keys = tuple(sorted(self.optimizers.keys()))
        optimizer_state_keys = tuple(sorted(optimizer_state.keys()))
        scheduler_keys = tuple(sorted(self.schedulers.keys()))
        scheduler_state_keys = tuple(sorted(scheduler_state.keys()))

        if optimizer_keys != optimizer_state_keys:
            idx = next(i for i in range(len(optimizer_keys)) if optimizer_keys[i] != optimizer_state_keys[i])
            raise Exception("Optimizer state mismatches: this optimizer has no attribute '{}'".format(optimizer_state_keys[idx]))
        
        if scheduler_keys != scheduler_state_keys:
            idx = next(i for i in range(len(scheduler_keys)) if scheduler_keys[i] != scheduler_state_keys[i])
            raise Exception("Scheduler state mismatches: this scheduler has no attribute '{}'".format(scheduler_state_keys[idx]))

        for k in self.optimizers:
            self.optimizers[k].load_state_dict(optimizer_state[k])
        for k in self.schedulers:
            self.schedulers[k].load_state_dict(scheduler_state[k])
