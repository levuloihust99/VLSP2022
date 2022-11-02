import torch
from torch import nn
from typing import Text

from torch.nn.init import xavier_uniform_
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertModel


class BertExtractive(torch.nn.Module):
    def __init__(self, encoder, inter_encoder, cls):
        super(BertExtractive, self).__init__()
        self.encoder = encoder
        self.inter_encoder = inter_encoder
        self.cls = cls
    

def create_model(cfg):
    config = BertConfig.from_pretrained(cfg.pretrained_encoder_model_path)
    if cfg.dropout is not None:
        config.attention_probs_dropout_prob = cfg.dropout
        config.hidden_dropout_prob = cfg.dropout
    encoder = BertModel.from_pretrained(cfg.pretrained_encoder_model_path, config=config)
    inter_encoder_config = BertConfig(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_attention_heads=config.num_attention_heads,
        num_hidden_layers=cfg.inter_encoder.num_hidden_layers
    )
    if cfg.dropout is not None:
        inter_encoder_config.attention_probs_dropout_prob = cfg.dropout
        inter_encoder_config.hidden_dropout_prob = cfg.dropout
    inter_encoder = BertEncoder(inter_encoder_config)
    for module in inter_encoder.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    cls = nn.Linear(encoder.config.hidden_size, 2)
    for p in cls.parameters():
        if p.dim() > 1:
            xavier_uniform_(p)
        else:
            p.data.zero_()
    model = BertExtractive(encoder=encoder, inter_encoder=inter_encoder, cls=cls)
    return model
