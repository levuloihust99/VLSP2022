from torch.optim.lr_scheduler import LambdaLR
from transformers import AdamW, get_linear_schedule_with_warmup


def get_optimizer(
    model,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.0,
    adam_epsilon: float = 1e-8,
    betas: tuple = (0.9, 0.999),
):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon, betas=betas)
    return optimizer


def get_schedule_linear(optimizer, warmup_steps, training_steps, steps_shift=0, last_epoch=-1):
    def lr_lambda(current_step):
        current_step += steps_shift
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0, float(training_steps - current_step) / float(max(1, training_steps - warmup_steps))
        )
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)
