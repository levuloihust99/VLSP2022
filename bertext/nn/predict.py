import re
import json
import argparse
import hydra
import torch

from tqdm import tqdm
from functools import partial
from underthesea import sent_tokenize, word_tokenize
from omegaconf import DictConfig, OmegaConf, open_dict

from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset

from .modeling import BertExtractive, create_model


class PredictDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


def collate_fn(items, tokenizer, max_length):
    num_sents = []
    max_seq_len = 0
    flatten_input_ids = []
    flatten_sentences = []
    for item in items:
        doc_sentences = item['sentences']
        flatten_sentences.extend(doc_sentences)
        doc_sentence_ids = item['sentence/input_ids']
        num_sents.append(len(doc_sentences))
        for input_ids in doc_sentence_ids:
            flatten_input_ids.append(input_ids)
            if max_seq_len < len(input_ids):
                max_seq_len = len(input_ids)
    
    max_seq_len = min(max_seq_len, max_length)
    flatten_input_ids_padded = []
    flatten_attn_mask = []
    # padding
    for input_ids in flatten_input_ids:
        input_ids = input_ids[:max_seq_len]
        input_ids[-1] = tokenizer.sep_token_id
        padding_len = max_seq_len - len(input_ids)
        attn_mask = [1] * len(input_ids)
        if padding_len > 0:
            input_ids += [tokenizer.pad_token_id] * padding_len
            attn_mask += [0] * padding_len
        flatten_input_ids_padded.append(input_ids)
        flatten_attn_mask.append(attn_mask)
    
    flatten_input_ids_padded = torch.tensor(flatten_input_ids_padded)
    flatten_attn_mask = torch.tensor(flatten_attn_mask)

    return {'sentences': flatten_sentences, 'sentence/input_ids': flatten_input_ids_padded, 
        'sentence/attn_mask': flatten_attn_mask, 'num_sents': num_sents}


def predict(model, dataloader, cfg, device):
    model.eval()
    all_predictions = []
    for batch in dataloader:
        sentence_input_ids = batch['sentence/input_ids'].to(device)
        sentence_attn_mask = batch['sentence/attn_mask'].to(device)

        # chunking
        chunk_size = cfg.chunk_size
        bsz = sentence_input_ids.size(0)
        num_chunks = (bsz - 1) // chunk_size + 1
        base_size = bsz // num_chunks
        remainder = bsz % num_chunks
        chunk_sizes = [base_size] * num_chunks
        added = [1] * remainder + [0] * (num_chunks - remainder)
        chunk_sizes = [c + a for c, a in zip(chunk_sizes, added)]

        chunked_sentence_input_ids = []
        chunked_sentence_attn_mask = []
        idx = 0
        for chunk_size in chunk_sizes:
            chunked_sentence_input_ids.append(sentence_input_ids[idx : idx + chunk_size])
            chunked_sentence_attn_mask.append(sentence_attn_mask[idx : idx + chunk_size])
        
        with torch.no_grad():
            all_sentence_embs = []
            for id_chunk, attn_chunk in zip(chunked_sentence_input_ids, chunked_sentence_attn_mask):
                chunked_outputs = model.encoder(input_ids=id_chunk, attention_mask=attn_chunk, return_dict=True)
                chunked_sequence_output = chunked_outputs.last_hidden_state
                chunked_pooled_output = chunked_sequence_output[:, 0, :]
                all_sentence_embs.append(chunked_pooled_output)
            all_sentence_embs = torch.cat(all_sentence_embs, dim=0)

            batch_sentence_embs = []
            batch_attn_mask = []
            idx = 0
            num_sents = batch['num_sents']
            max_num_sent = max(num_sents)
            for num_sent in num_sents:
                per_doc_sent_embs = all_sentence_embs[idx : idx + num_sent]
                padding_len = max_num_sent - num_sent
                attn_mask = [1] * num_sent
                if padding_len > 0:
                    per_doc_sent_embs = torch.cat(
                        [per_doc_sent_embs, torch.zeros(padding_len, per_doc_sent_embs.size(-1))],
                        dim=0
                    )
                    attn_mask += [0] * (max_num_sent - num_sent)
                attn_mask = torch.tensor(attn_mask).to(device)
                batch_sentence_embs.append(per_doc_sent_embs)
                batch_attn_mask.append(attn_mask)
            
            # inter encoder
            extended_batch_attn_mask = batch_attn_mask[:, None, None, :]
            extended_batch_attn_mask = (1.0 - extended_batch_attn_mask)  * -10000.0
            inter_outputs = model.inter_encoder(hidden_states=batch_sentence_embs,
                attention_mask=extended_batch_attn_mask)
            inter_sent_embs = inter_outputs.last_hidden_state
            active_mask = batch_attn_mask.view(-1).to(torch.bool)
            flatten_inter_sent_embs = inter_sent_embs.view(-1, inter_sent_embs.size(-1))[active_mask]
            flatten_inter_sent_logits = model.cls(flatten_inter_sent_embs)
            predictions = torch.argmax(flatten_inter_sent_logits, dim=-1)

            batch_predictions = []
            batch_sentences = []
            idx = 0
            for num_sent in num_sents:
                batch_predictions.append(predictions[idx : idx + num_sent])
                batch_sentences.append(batch['sentences'][idx : idx + num_sent])
                idx += num_sent
            
            final_preds = []
            for idx, pred in enumerate(batch_predictions):
                picked_sent_ids = pred.nonzero.view(-1)
                picked_sents = [batch_sentences[idx][_id] for _id in picked_sent_ids]
                final_preds.append(" ".join(picked_sents))

            all_predictions.extend(final_preds)

        return all_predictions
    

@hydra.main(version_base=None, config_path="../conf", config_name="predict_conf")
def main(cfg: DictConfig):
    data = []
    with open(cfg.data_path, "r") as reader:
        for line in reader:
            data.append(json.loads(line.strip()))

    all_sents = []
    for item in tqdm(data):
        doc_sents = []
        for doc in item['single_documents']:
            sents = sent_tokenize(doc['raw_text'])
            extended_sents = []
            for sent in sents:
                sent = re.sub(r"\n\s*", "\n", sent)
                splits = sent.split('\n')
                extended_sents.extend(splits)
            doc_sents.extend(extended_sents)
        all_sents.append(doc_sents)
    
    # tokenizer
    tokenizer = BertTokenizer.from_pretrained(cfg.tokenizer_path)

    tokenized_data = []
    for doc_sents in tqdm(all_sents):
        doc_sentence_ids = []
        for sent in doc_sents:
            input_ids = tokenizer(sent).input_ids
            doc_sentence_ids.append(input_ids)
        tokenized_data.append(doc_sentence_ids)
    
    compact_data = []
    for idx in range(len(all_sents)):
        doc_sents = all_sents[idx]
        doc_sentence_ids = tokenized_data[idx]
        compact_data.append({
            'sentences': doc_sents,
            'sentence/input_ids': doc_sentence_ids
        })
    
    dataset = PredictDataset(tokenized_data)
    wrapped_collate_fn = partial(collate_fn, tokenizer=tokenizer, max_length=cfg.max_length)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=wrapped_collate_fn, shuffle=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = create_model(cfg)
    model.to(device)

    all_predictions = predict(model, dataloader, cfg, device)
    with open(cfg.output_path, "w") as writer:
        for pred_sent in all_predictions:
            writer.write(pred_sent + "\n")


if __name__ == "__main__":
    main()