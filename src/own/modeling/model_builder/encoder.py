import torch

from typing import Literal, Text, Optional

from transformers import BertConfig, BertModel


def init_encoder(
    architecture: Literal["bert", "primera", "roberta"],
    pretrained_model_path: Optional[Text] = None,
    hidden_size: int = 768,
    intermediate_hidden_size: int = 3072,
    num_hidden_layers: int = 12,
    max_encoder_sequence_length: int = 512,
    **kwargs
):
    if architecture == 'bert':
        init_fn = init_bert_encoder
    elif architecture == 'primera':
        init_fn = init_primera_encoder
    elif architecture == 'roberta':
        init_fn = init_roberta_encoder
    else:
        raise Exception("Architecture '{}' is not supported. \
            Available options are: 'bert', 'primera', 'roberta'".format(architecture))
    
    return init_fn(
        pretrained_model_path=pretrained_model_path,
        hidden_size=hidden_size,
        intermediate_hidden_size=intermediate_hidden_size,
        num_hidden_layers=num_hidden_layers,
        max_encoder_sequence_length=max_encoder_sequence_length,
        **kwargs
    )


def init_bert_encoder(
    pretrained_model_path: Optional[Text] = None,
    hidden_size: int = 768,
    intermediate_hidden_size: int = 3072,
    num_hidden_layers: int = 12,
    max_encoder_sequence_length: int = 512,
    **kwargs
):
    if pretrained_model_path:
        encoder = BertModel.from_pretrained(pretrained_model_path, add_pooling_layer=False)
        encoder_config = encoder.config
    else:
        encoder_config = BertConfig(
            hidden_size=hidden_size,
            intermediate_hidden_size=intermediate_hidden_size,
            num_hidden_layers=num_hidden_layers
        )
        encoder = BertModel(encoder_config, add_pooling_layer=False)
    if encoder.embeddings.position_embeddings.weight.size(0) < max_encoder_sequence_length:
        updated_position_embeddings = torch.nn.Embedding(max_encoder_sequence_length, encoder_config.hidden_size)
        current_max_length = encoder.embeddings.position_embeddings.weight.size(0)
        updated_position_embeddings.weight.data[:current_max_length] = \
            encoder.embeddings.position_embeddings.weight.data
        added_length = max_encoder_sequence_length - current_max_length
        updated_position_embeddings.weight.data[current_max_length:] = \
            encoder.embeddings.position_embeddings.weight.data[-1:].repeat(added_length, 1)
        encoder.embeddings.position_embeddings = updated_position_embeddings
        encoder.embeddings.position_ids = torch.arange(max_encoder_sequence_length).unsqueeze(0)
        
    return encoder


def init_primera_encoder(
    pretrained_model_path: Optional[Text] = None,
    hidden_size: int = 768,
    intermediate_hidden_size: int = 3072,
    num_hidden_layers: int = 12,
    **kwargs
):
    pass


def init_roberta_encoder(

):
    pass
