import torch
from src.own.modeling.optimization import create_optimizers_and_schedulers
from transformers import BertModel


model = BertModel.from_pretrained("bert-base-uncased")
optim, scheduler = create_optimizers_and_schedulers(model, total_steps=10000, weight_decay=0.1,
    learning_rate=2e-5)

input_ids = torch.tensor([[1, 2, 3, 4, 5]])
output = model(input_ids=input_ids, return_dict=True)
sequence_output = output.last_hidden_state
pooled_output = sequence_output[:, 0, :]
pooled_output.sum().backward()

optim.step()