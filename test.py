import torch
from transformers import BertConfig, BertModel
from src.own.modeling.modeling_utils import recursive_apply

encoder = BertModel(BertConfig())
decoder = BertModel(BertConfig(is_decoder=True, add_cross_attention=True))
encoder_input_ids = torch.tensor([[1, 2, 3, 4, 5]])
decoder_input_ids = torch.tensor([[1, 2, 3]])

encoder_outputs = encoder(
    input_ids=encoder_input_ids,
    return_dict=True
)
encoder_hidden_states = encoder_outputs.last_hidden_state
decoder_outputs = decoder(
    input_ids=decoder_input_ids,
    encoder_hidden_states=encoder_hidden_states,
    use_cache=True,
    return_dict=True
)

past_key_values = decoder_outputs.past_key_values
after = recursive_apply(past_key_values, fn=lambda x: x.reshape(-1))
print("done")