import pickle
import argparse
import torch
from torch import nn
from typing import Optional, List
from collections import namedtuple

from transformers import BertModel, BertConfig, BertTokenizer
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

from .summarizer import AbsSummarizer


TransformersDecoderOutput = namedtuple(
    "TransformersDecoderOutput", ["logits", "past_key_values"]
)


class TransformersDecoder(nn.Module):
    def __init__(self, backbone, head):
        """Create a transformers decoder model.

        Args:
            backbone: A stack of transformer layer with cross attention sub-layer.
            generator: A language model head
        """
        super(TransformersDecoder, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        sequence_output = outputs.last_hidden_state
        logits = self.head(sequence_output)
        return TransformersDecoderOutput(logits=logits, past_key_values=outputs.past_key_values)


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder-path', default='checkpoints/encoder.pt')
    parser.add_argument('--decoder-path', default='checkpoints/decoder.pt')
    args = parser.parse_args()

    # < encoder: initialization
    encoder = BertModel(BertConfig())
    encoder_saved_state = torch.load(args.encoder_path, map_location=lambda s, t: s)
    encoder.load_state_dict(encoder_saved_state)
    encoder.eval()
    # encoder />

    # < decoder: initialization
    backbone = BertModel(BertConfig(is_decoder=True, add_cross_attention=True))
    head = BertOnlyMLMHead(BertConfig())
    decoder = TransformersDecoder(backbone, head)
    decoder_saved_state = torch.load(args.decoder_path, map_location=lambda s, t: s)
    decoder.load_state_dict(decoder_saved_state)
    decoder.eval()
    # decoder />

    # < encoder: calculate encoder hidden states
    # encoder_input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    # outputs = encoder(input_ids=encoder_input_ids, return_dict=True)
    # encoder_hidden_states = outputs.last_hidden_state
    # encoder />
    
    # < decoder
    # input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    # output_ids = input_ids.clone()
    # max_length = 20
    # cur_length = input_ids.size(1)
    # past_key_values = None
    # while cur_length < max_length:
    #     outputs = decoder(
    #         input_ids=input_ids,
    #         encoder_hidden_states=encoder_hidden_states,
    #         past_key_values=past_key_values,
    #         use_cache=True
    #     )
    #     past_key_values = outputs.past_key_values
    #     logits = outputs.logits
    #     input_ids = torch.argmax(logits, dim=-1)[:, -1:]
    #     output_ids = torch.cat([output_ids, input_ids], dim=1)
    #     cur_length = output_ids.size(1)
    # decoder />

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # < AbsSummarizer
    abs_summarizer = AbsSummarizer(
        encoder=encoder,
        decoder=decoder,
        tokenizer=tokenizer,
    )
    # AbsSummarizer />
    abs_summarizer

if __name__ == "__main__":
    # main()
    test()
