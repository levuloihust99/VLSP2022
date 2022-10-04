import torch

from typing import Text
from .datasets import ByteDataset
from .sampler import ReproducibleRandomSampler

from torch.utils.data import DataLoader


def create_dataloader(
    data_path: Text,
    batch_size: int,
    max_encoder_sequence_length: int,
    encoder_sep_token_id: int,
    encoder_pad_token_id: int,
    max_decoder_sequence_length: int,
    decoder_end_token_id: int,
    decoder_pad_token_id: int,
    use_segmentation: bool = True,
    training: bool = True
):
    dataset = ByteDataset(data_path, idx_record_size=6)
    if training:
        sampler = ReproducibleRandomSampler(dataset)
        collate_fn = get_collate_fn(
            max_encoder_sequence_length=max_encoder_sequence_length,
            encoder_sep_token_id=encoder_sep_token_id,
            encoder_pad_token_id=encoder_pad_token_id,
            max_decoder_sequence_length=max_decoder_sequence_length,
            decoder_end_token_id=decoder_end_token_id,
            decoder_pad_token_id=decoder_pad_token_id,
            use_segmentation=use_segmentation
        )
        data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
    else:
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return data_loader


def get_collate_fn(
    max_encoder_sequence_length: int,
    encoder_sep_token_id: int,
    encoder_pad_token_id: int,
    max_decoder_sequence_length: int,
    decoder_end_token_id: int,
    decoder_pad_token_id: int,
    use_segmentation: bool = True
):
    def collate_fn(items):
        max_encoder_sequence_length_in_batch = 0
        max_decoder_sequence_length_in_batch = 0

        for item in items:
            if max_encoder_sequence_length_in_batch < len(item['src']):
                max_encoder_sequence_length_in_batch = len(item['src'])
            if max_decoder_sequence_length_in_batch < len(item['tgt']) - 1:
                max_decoder_sequence_length_in_batch = len(item['tgt']) - 1
        
        max_encoder_sequence_length_in_batch = min(max_encoder_sequence_length,
                                        max_encoder_sequence_length_in_batch)
        max_decoder_sequence_length_in_batch = min(max_decoder_sequence_length,
                                        max_decoder_sequence_length_in_batch)

        batch = {
            'encoder_input_ids': [],
            'encoder_attention_mask': [],
            'encoder_token_type_ids': [],
            'decoder_input_ids': [],
            'decoder_attention_mask': [],
            'labels': []
        }
        for item in items:
            output_item = {}
            output_item['encoder_input_ids'] = item['src'][:max_encoder_sequence_length_in_batch][:-1] \
                                                + [encoder_sep_token_id]
            output_item['encoder_attention_mask'] = [1] * len(output_item['encoder_input_ids'])
            if use_segmentation:
                output_item['encoder_token_type_ids'] = item['segs'][:max_encoder_sequence_length_in_batch]
            else:
                output_item['encoder_token_type_ids'] = [0] * len(output_item['encoder_input_ids'])
            if len(output_item['encoder_input_ids']) < max_encoder_sequence_length_in_batch:
                padding_length = max_encoder_sequence_length_in_batch - len(output_item['encoder_input_ids'])
                output_item['encoder_input_ids'] += [encoder_pad_token_id] * padding_length
                output_item['encoder_attention_mask'] += [0] * padding_length
                output_item['encoder_token_type_ids'] += [0] * padding_length
            
            output_item['decoder_input_ids'] = item['tgt'][:-1][:max_decoder_sequence_length_in_batch]
            output_item['labels'] = item['tgt'][1:][:max_decoder_sequence_length_in_batch][:-1] + [decoder_end_token_id]
            output_item['decoder_attention_mask'] = [1] * len(output_item['decoder_input_ids'])
            if len(output_item['decoder_input_ids']) < max_decoder_sequence_length_in_batch:
                padding_length = max_decoder_sequence_length_in_batch - len(output_item['decoder_input_ids'])
                output_item['decoder_input_ids'] += [decoder_pad_token_id] * padding_length
                output_item['labels'] += [-1] * padding_length
                output_item['decoder_attention_mask'] += [0] * padding_length
            
            batch['encoder_input_ids'].append(output_item['encoder_input_ids'])
            batch['encoder_attention_mask'].append(output_item['encoder_attention_mask'])
            batch['encoder_token_type_ids'].append(output_item['encoder_token_type_ids'])
            batch['decoder_input_ids'].append(output_item['decoder_input_ids'])
            batch['decoder_attention_mask'].append(output_item['decoder_attention_mask'])
            batch['labels'].append(output_item['labels'])
        
        batch = {k: torch.tensor(v) for k, v in batch.items()}
        return batch

    return collate_fn
