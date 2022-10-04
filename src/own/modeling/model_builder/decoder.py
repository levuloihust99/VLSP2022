import re
import torch
import copy

from typing import Literal
from transformers import BertForMaskedLM, BertConfig

from ..decoder import TransformersDecoder


def init_decoder(
    architecture: Literal["bert", "primera", "roberta"],
    hidden_size: int = 768,
    intermediate_hidden_size: int = 3072,
    num_hidden_layers: int = 6,
    max_decoder_sequence_length: int = 512,
    **kwargs
):
    if architecture == 'bert':
        init_fn = init_bert_decoder
    elif architecture == 'primera':
        init_fn = init_primera_decoder
    elif architecture == 'roberta':
        init_fn = init_roberta_decoder
    
    return init_fn(
        hidden_size=hidden_size,
        intermediate_hidden_size=intermediate_hidden_size,
        num_hidden_layers=num_hidden_layers,
        max_decoder_sequence_length=max_decoder_sequence_length,
        **kwargs
    )


def init_bert_decoder(
    pretrained_model_path,
    hidden_size: int = 768,
    intermediate_hidden_size: int = 3072,
    num_hidden_layers: int = 6,
    max_decoder_sequence_length: int = 512,
    **kwargs
):
    if pretrained_model_path:
        pretrained_decoder = BertForMaskedLM.from_pretrained(pretrained_model_path,
            is_decoder=True, add_cross_attention=True)
        decoder_config = pretrained_decoder.config
        
        if pretrained_decoder.config.num_hidden_layers != num_hidden_layers:
            # < inject weights
            pretrained_decoder_state_dict = pretrained_decoder.state_dict()
            intermediate_decoder_config = copy.deepcopy(decoder_config)
            intermediate_decoder_config.num_hidden_layers = num_hidden_layers
            intermediate_decoder = BertForMaskedLM(intermediate_decoder_config)
            intermediate_decoder_state_dict = intermediate_decoder.state_dict()
            shifted = max(decoder_config.num_hidden_layers - num_hidden_layers, 0)
            for k in intermediate_decoder_state_dict:
                if k.startswith('bert.encoder.layer'):
                    layer_num = re.search(r'bert\.encoder\.layer\.(\d+)', k)
                    layer_num = int(layer_num.group(1))
                    shifted_layer_num = layer_num + shifted
                    shifted_key = re.sub(r'(bert\.encoder\.layer\.)(\d+)(.*)', rf'\g<1>{shifted_layer_num}\g<3>', k)
                    intermediate_decoder_state_dict[k] = pretrained_decoder_state_dict[shifted_key]
            intermediate_decoder.load_state_dict(intermediate_decoder_state_dict)
            # inject weights />
            backbone = intermediate_decoder.bert
            head = intermediate_decoder.cls
        else:
            backbone = pretrained_decoder.bert
            head = pretrained_decoder.cls
        decoder = TransformersDecoder(backbone=backbone, head=head)
    else:
        decoder_config = BertConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_hidden_size,
            num_hidden_layers=num_hidden_layers,
            is_decoder=True, add_cross_attention=True, **kwargs
        )
        base_decoder = BertForMaskedLM(decoder_config)
        decoder = TransformersDecoder(backbone=base_decoder.bert, head=base_decoder.cls)
    
    if decoder.backbone.embeddings.position_embeddings.weight.size(0) < max_decoder_sequence_length:
        updated_position_embeddings = torch.nn.Embedding(max_decoder_sequence_length, decoder_config.hidden_size)
        current_max_length = decoder.backbone.embeddings.position_embeddings.weight.size(0)
        updated_position_embeddings.weight.data[:current_max_length] = \
            decoder.backbone.embeddings.position_embeddings.weight.data
        added_length = max_decoder_sequence_length - current_max_length
        updated_position_embeddings.weight.data[current_max_length:] = \
            decoder.backbone.embeddings.position_embeddings.weight.data[-1:].repeat(added_length, 1)
        decoder.backbone.embeddings.position_embeddings = updated_position_embeddings
        decoder.backbone.embeddings.position_ids = torch.arange(max_decoder_sequence_length).unsqueeze(0)
    return decoder


def init_primera_decoder(**kwargs):
    pass


def init_roberta_decoder(**kwargs):
    pass
