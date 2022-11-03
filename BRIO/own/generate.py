import json
import argparse
import torch
import logging

from tqdm import tqdm
from typing import Optional, Tuple

from .utils import tile, recursive_apply


def greedy_multiple(
    model,
    tokenizer,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    beam_size: int = 1,
    min_length: int = 0,
    max_length: int = 100,
    alpha: float = 0.6,
    block_trigram: bool = True,
    num_return_sequences: int = 1,
    do_sample: bool = False,
    num_sampling: int = 32,
    max_candidates_per_beam: int = 0
):
    # 1. initialization
    decoder_end_token_id = tokenizer.eos_token_id
    topk_log_probs = torch.tensor([0.0] + [float("-inf")] * (num_return_sequences - 1)).to(model.device) # [expanded_beam_size]
    alive_seq = torch.full(
        [num_return_sequences, 1],
        model.config.decoder_start_token_id,
        dtype=torch.long,
        device=model.device
    )

    if encoder_hidden_states is None:
        encoder_outputs = model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state # [1, seq_len, hidden_size]
    encoder_hidden_states_tiled = tile(encoder_hidden_states, num_return_sequences, dim=0) # [B, seq_len, hidden_size]
    past_key_values = None

    # decode first step
    decoder_input_ids = alive_seq[:, -1:] # [B, 1]
    decoder_outputs = model.decoder(
        input_ids=decoder_input_ids,
        encoder_hidden_states=encoder_hidden_states_tiled,
        use_cache=True,
        past_key_values=past_key_values
    )
    past_key_values = decoder_outputs.past_key_values
    sequence_output = decoder_outputs[0] # [B, 1, vocab_size]
    if model.config.tie_word_embeddings:
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (model.model_dim**-0.5)
    logits = model.lm_head(sequence_output)
    vocab_size = logits.size(-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).squeeze(1) # [B, vocab_size]
    log_probs[:, decoder_end_token_id] = -1e20

    # sum of log probs of the current sequence
    log_probs += topk_log_probs.unsqueeze(1) # [B, vocab_size]

    cur_scores = log_probs / 1.0 # [B, vocab_size]
    cur_scores = cur_scores.view(-1) # [B * vocab_size]
    topk_scores, topk_ids = cur_scores.topk(num_return_sequences, dim=0)
    
    alive_seq = torch.cat([alive_seq, torch.unsqueeze(topk_ids, dim=1)], dim=1)

    hypotheses = []
    # 3. multi greedy
    for idx in tqdm(range(num_return_sequences), total=num_return_sequences, desc="Candidate"):
        seq = alive_seq[idx: idx + 1] # [1, 2]
        past_k_v = recursive_apply(past_key_values, lambda x: x.index_select(0, torch.tensor([0]).to(model.device)))
        for step in tqdm(range(max_length), total=max_length, desc="Step"):
            decoder_input_ids = seq[:, -1:] # [1, 1]
            decoder_outputs = model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                use_cache=True,
                past_key_values=past_k_v
            )
            past_k_v = decoder_outputs.past_key_values
            sequence_output = decoder_outputs[0]
            if model.config.tie_word_embeddings:
                # Rescale output before projecting on vocab
                # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
                sequence_output = sequence_output * (model.model_dim**-0.5)
            logits = model.lm_head(sequence_output) # [1, 1, vocab_size]
            next_id = torch.argmax(logits, dim=-1) # [1, 1]
            seq = torch.cat([seq, next_id], dim=1)
            if int(next_id[0][0]) == decoder_end_token_id:
                break
        with tokenizer.as_target_tokenizer():
            hypotheses.append(tokenizer.decode(seq[0], clean_up_tokenization_spaces=False, skip_special_tokens=True))
    return hypotheses


def generate(
    model,
    tokenizer,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    beam_size: int = 1,
    min_length: int = 0,
    max_length: int = 100,
    alpha: float = 0.6,
    block_trigram: bool = True,
    num_return_sequences: int = 1,
    do_sample: bool = False,
    num_sampling: int = 32,
    max_candidates_per_beam: int = 0
):
    expanded_beam_size = beam_size * 3
    decoder_end_token_id = tokenizer.eos_token_id
    if input_ids is None:
        assert inputs_embeds is not None, \
            "At least 'input_ids' or 'inputs_embeds' must be provided."

    if encoder_hidden_states is None:
        encoder_outputs = model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state # [1, seq_len, hidden_size]
    encoder_hidden_states = tile(encoder_hidden_states, expanded_beam_size, dim=0) # [expanded_beam_size, seq_len, hidden_size]
    
    hypotheses = []
    alive_seq = torch.full(
        [expanded_beam_size, 1],
        model.config.decoder_start_token_id,
        dtype=torch.long,
        device=model.device
    )
    topk_log_probs = torch.tensor([0.0] + [float("-inf")] * (expanded_beam_size - 1)).to(model.device) # [expanded_beam_size]
    topk_scores = torch.empty_like(topk_log_probs).copy_(topk_log_probs).to(model.device)
    past_key_values = None

    for step in range(max_length):
        # 1. if encounter repeated trigram, ignore
        repeated_trigram_beams = []
        if block_trigram:
            cur_length = alive_seq.size(1)
            if cur_length > 3:
                for idx in range(alive_seq.size(0)):
                    fail = False
                    words = [int(w) for w in alive_seq[idx]]
                    words = tokenizer.decode(words).split()
                    if len(words) <=3:
                        continue
                    trigrams = [(words[i - 1], words[i], words[i + 1]) for i in range(1, len(words) - 1)]
                    trigram = tuple(trigrams[-1])
                    if trigram in trigrams[:-1]:
                        fail = True
                    if fail:
                        cur_scores[idx] = -1e20
                        repeated_trigram_beams.append(idx)
        kept_beams = [idx for idx in range(alive_seq.size(0)) if idx not in repeated_trigram_beams]
        kept_beams = torch.tensor(kept_beams).to(model.device)
        alive_seq = alive_seq.index_select(0, kept_beams)
        if past_key_values:
            past_key_values = recursive_apply(past_key_values, lambda x: x.index_select(0, kept_beams))
        encoder_hidden_states = encoder_hidden_states.index_select(0, kept_beams)
        topk_log_probs = topk_log_probs.index_select(0, kept_beams)
        
        # 2. if beam ends, add to hypotheses and filter beams
        ended_beams = []
        for idx in range(alive_seq.size(0)):
            prev_generated_token_id = alive_seq[idx][-1]
            if prev_generated_token_id == decoder_end_token_id:
                ended_beams.append(idx)
                hypotheses.append({'sequence': alive_seq[idx], 'score': topk_scores[idx]})
        non_ended_beams = [idx for idx in range(alive_seq.size(0)) if idx not in ended_beams]
        non_ended_beams = torch.tensor(non_ended_beams).to(model.device)
        alive_seq = alive_seq.index_select(0, non_ended_beams)
        if past_key_values:
            past_key_values = recursive_apply(past_key_values, lambda x: x.index_select(0, non_ended_beams))
        encoder_hidden_states = encoder_hidden_states.index_select(0, non_ended_beams)
        topk_log_probs = topk_log_probs.index_select(0, non_ended_beams)

        if alive_seq.size(0) == 0:
            break

        # 3. generate
        decoder_input_ids = alive_seq[:, -1:] # [B, 1]
        decoder_outputs = model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=True,
            past_key_values=past_key_values
        )
        past_key_values = decoder_outputs.past_key_values
        sequence_output = decoder_outputs[0] # [B, 1, vocab_size]
        if model.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (model.model_dim**-0.5)
        logits = model.lm_head(sequence_output)
        vocab_size = logits.size(-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1).squeeze(1) # [B, vocab_size]
        if step < min_length:
            log_probs[:, decoder_end_token_id] = -1e20
        
        # sum of log probs of the current sequence
        log_probs += topk_log_probs.unsqueeze(1) # [B, vocab_size]

        length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
        cur_scores = log_probs / length_penalty # [B, vocab_size]
        cur_scores = cur_scores.view(-1) # [B * vocab_size]

        # size of topk_scores is always equal to size 0 of alive_seq
        if step == 0:
            topk_scores, topk_ids = cur_scores.topk(expanded_beam_size, dim=0) # [B]
        else:
            topk_scores = []
            topk_ids = []
            for idx in range(alive_seq.size(0)):
                beam_vocab_score = cur_scores[idx]
                beam_topk_scores, beam_topk_ids = beam_vocab_score.topk(1, dim=0) # [1]
                topk_scores.append(beam_topk_scores)
                topk_ids.append(beam_topk_ids)
            topk_scores = torch.cat(topk_scores, dim=0).to(model.device)
            topk_ids = torch.cat(topk_ids, dim=0).to(model.device)

        alive_seq = torch.cat([alive_seq, torch.unsqueeze(topk_ids, dim=1)], dim=1)
        topk_log_probs = topk_scores * length_penalty
    
    if alive_seq.size(0) > 0:
        hypotheses.extend([{'sequence': alive_seq[idx], 'score': topk_scores[idx]} for idx in range(alive_seq.size(0))])
    hypotheses.sort(key=lambda x: x['score'], reverse=True)
    return hypotheses[:num_return_sequences]
    

def batch_generate(
    model,
    tokenizer,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    beam_size: int = 1,
    min_length: int = 0,
    max_length: int = 100,
    alpha: float = 0.6,
    block_trigram: bool = True,
    num_return_sequences: int = 1,
    do_sample: bool = False,
    num_sampling: int = 32,
    max_candidates_per_beam: int = 0
):
    """Decoding process stops either when the output sequence length exceeds `max_length` or `num_return_sequences` hypotheses is reached."""
    if max_candidates_per_beam == 0:
        max_candidates_per_beam = beam_size
    decoder_end_token_id = tokenizer.eos_token_id
    if input_ids is None:
        assert inputs_embeds is not None, \
            "At least 'input_ids' or 'inputs_embeds' must be provided."
        batch_size = inputs_embeds.size(0)
    else:
        batch_size = input_ids.size(0)

    if encoder_hidden_states is None:
        encoder_outputs = model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            return_dict=True
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state
    encoder_hidden_states = tile(encoder_hidden_states, beam_size, dim=0)

    # Structure that holds finished hypotheses
    hypotheses = [[] for _ in range(batch_size)]

    batch_offset = torch.arange(
        batch_size, dtype=torch.long, device=model.device)
    beam_offset = torch.arange(
        0,
        batch_size * beam_size,
        step=beam_size,
        dtype=torch.long,
        device=model.device)
    alive_seq = torch.full(
        [batch_size * beam_size, 1],
        model.config.decoder_start_token_id,
        dtype=torch.long,
        device=model.device
    )

    # Give full probability to the first beam on the first step.
    topk_log_probs = torch.tensor([0.0] + [float("-inf")] * (beam_size - 1)).repeat(batch_size).to(model.device) # [batch_size * beam_size]

    past_key_values = None
    maximum_num_per_beam = 0
    for step in range(max_length):
        decoder_input_ids = alive_seq[:, -1:] # [batch_size * beam_size, 1]
        decoder_outputs = model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=True,
            past_key_values=past_key_values
        )
        past_key_values = decoder_outputs.past_key_values
        sequence_output = decoder_outputs[0] # [batch_size * beam_size, 1, vocab_size]
        if model.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (model.model_dim**-0.5)
        logits = model.lm_head(sequence_output)
        vocab_size = logits.size(-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1).squeeze(1) # [batch_size * beam_size, vocab_size]

        if step < min_length:
            log_probs[:, decoder_end_token_id] = -1e20
        
        # sum of log probs of the current sequence
        log_probs += topk_log_probs.view(-1).unsqueeze(1) # [batch_size * beam_size, vocab_size]

        length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
        cur_scores = log_probs / length_penalty # [batch_size * beam_size, vocab_size]

        ignored_beams = set()
        # if trigram is repeated in a beam, ignore that beam (assign large minus score)
        if block_trigram:
            cur_length = alive_seq.size(1)
            if cur_length > 3:
                for idx in range(alive_seq.size(0)):
                    fail = False
                    words = [int(w) for w in alive_seq[idx]]
                    words = tokenizer.decode(words).split()
                    if len(words) <=3:
                        continue
                    trigrams = [(words[i - 1], words[i], words[i + 1]) for i in range(1, len(words) - 1)]
                    trigram = tuple(trigrams[-1])
                    if trigram in trigrams[:-1]:
                        fail = True
                    if fail:
                        cur_scores[idx] = -1e20
                        ignored_beams.add(idx)
        # ignore beams that ended
        for idx in range(alive_seq.size(0)):
            prev_generated_token_id = alive_seq[idx][-1]
            if prev_generated_token_id == decoder_end_token_id:
                cur_scores[idx] = -1e20 # ignore beam `idx`
                ignored_beams.add(idx)
        if step == 0:
            non_top_beams_mask = torch.tensor([False] + [True] * (beam_size - 1)).unsqueeze(0).repeat(batch_size, 1).view(-1).to(model.device)
            non_top_beams = torch.arange(beam_size * batch_size)[non_top_beams_mask].tolist()
            ignored_beams = ignored_beams.union(set(non_top_beams))

        # track ignored beams
        ignored_beams = torch.tensor(sorted(list(ignored_beams)))
        ignored_batch_of_beams = ignored_beams.div(beam_size, rounding_mode="trunc").tolist()
        ignored_beams_in_batch = ignored_beams.fmod(beam_size).tolist()
        ignored = {k: set() for k in ignored_batch_of_beams}
        for batch, beam in zip(ignored_batch_of_beams, ignored_beams_in_batch):
            ignored[batch].add(beam)
        ignored = {k: sorted(list(v)) for k, v in ignored.items()}
        for k in range(batch_offset.size(0)): # current batch size
            if k not in ignored:
                ignored[k] = []

        cur_scores = cur_scores.reshape(-1, beam_size * vocab_size) # [batch_size, beam_size * vocab_size]
        if do_sample and num_sampling > beam_size:
            top_scores, top_ids = cur_scores.topk(num_sampling, dim=-1) # [batch_size, num_sampling]
            choices = torch.randperm(num_sampling)[:beam_size].to(model.device) # [beam_size]
            topk_scores = top_scores.index_select(1, choices) # [batch_size, beam_size]
            topk_ids = top_ids.index_select(1, choices) # [batch_size, beam_size]
            idxs = torch.argsort(topk_scores, dim=1, descending=True)
            topk_scores = torch.gather(topk_scores, 1, idxs)
            topk_ids = torch.gather(topk_ids, 1, idxs)
        else:
            cur_scores_3d = cur_scores.view(-1, beam_size, vocab_size)
            topk_scores = []
            topk_ids = []
            for batch_idx in range(cur_scores_3d.size(0)):
                batch_cur_scores = cur_scores_3d[batch_idx] # [beam_size, vocab_size]
                num_non_ignored_beams = beam_size - len(ignored[batch_idx])
                if num_non_ignored_beams == 0:
                    logger.info("Encounter a batch with all beams are ignored.")
                    ignored[batch_idx] = []
                    num_per_beam = max_candidates_per_beam
                else:
                    num_per_beam = (beam_size - 1) // num_non_ignored_beams + 1
                num_per_beam = max(num_per_beam, max_candidates_per_beam)
                if maximum_num_per_beam < num_per_beam and step > 0:
                    maximum_num_per_beam = num_per_beam
                batch_topk_scores = []
                batch_topk_ids = []
                for beam_idx, beam_scores in enumerate(batch_cur_scores): # beam_scrores: [vocab_size]
                    if beam_idx in ignored[batch_idx]:
                        continue
                    else:
                        beam_topk_scores, beam_topk_ids = beam_scores.topk(num_per_beam) # [num_per_beam]
                        batch_topk_scores.append(beam_topk_scores)
                        batch_topk_ids.append(beam_topk_ids + beam_idx * vocab_size)
                batch_topk_scores = torch.cat(batch_topk_scores, dim=0) # [num_per_beam * num_non_ignored_beams]
                batch_topk_ids = torch.cat(batch_topk_ids, dim=0) # [num_per_beam * num_non_ignored_beams]

                _topk_scores, _topk_ids = batch_topk_scores.topk(beam_size, dim=0) # [beam_size]
                _topk_ids = torch.gather(batch_topk_ids, 0, _topk_ids)
                topk_scores.append(_topk_scores)
                topk_ids.append(_topk_ids)
            topk_scores = torch.stack(topk_scores, dim=0) # [batch_size, beam_size]
            topk_ids = torch.stack(topk_ids, dim=0) # [batch_size, beam_size]
        
        # Recover log probs
        topk_log_probs = topk_scores * length_penalty

        # Resolve beam origin and true word ids
        topk_beam_index = topk_ids.div(vocab_size, rounding_mode="trunc")
        topk_ids = topk_ids.fmod(vocab_size)

        # Map beam_index to batch_index in the flat representation
        batch_index = (
            topk_beam_index
            + beam_offset.unsqueeze(1)
        ) # [batch_size, beam_size]
        select_indices = batch_index.view(-1) # [batch_size * beam_size]

        # Append last prediction
        alive_seq = torch.cat(
            [alive_seq.index_select(0, select_indices),
            topk_ids.view(-1, 1)],
            dim=-1
        )

        is_finished = topk_ids.eq(decoder_end_token_id) # [batch_size, beam_size]
        if step + 1 == max_length:
            is_finished.fill_(1)
        # End condition is top beam is finished and number of hypotheses is greater than or equal to num_return_sequences
        end_condition = is_finished[:, 0].eq(1) # [batch_size]: top beam is finished
        reach_num_return_sequences = torch.tensor(
            [len(hypotheses[batch_offset[idx]]) >= num_return_sequences for idx in range(batch_offset.size(0))]
        ).to(model.device)
        end_condition = end_condition & torch.tensor(reach_num_return_sequences)
        # Save finished hypotheses
        if is_finished.any():
            predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1)) # [batch_size, beam_size, seq_length]
            for batch_idx in range(batch_size):
                b = batch_offset[batch_idx]
                if end_condition[batch_idx]: # top beam of batch idx is finished
                    is_finished[batch_idx].fill_(1)
                finished_hyp = is_finished[batch_idx].nonzero().view(-1) # List of beam_idx of batch `batch_idx` at which hypothesis ends.
                # Store finished hypotheses for this batch
                for beam_idx in finished_hyp:
                    hypotheses[b].append((
                        topk_scores[batch_idx, beam_idx],
                        predictions[batch_idx, beam_idx, 1:]
                    ))
            
            batches_non_finished = end_condition.eq(0).nonzero().view(-1)
            # If all sentences are translated, no need to go further
            if len(batches_non_finished) == 0:
                break
            # Remove finished batches for the next step.
            topk_log_probs = topk_log_probs.index_select(0, batches_non_finished)
            batch_index = batch_index.index_select(0, batches_non_finished)
            batch_offset = batch_offset.index_select(0, batches_non_finished)
            alive_seq = predictions.index_select(0, batches_non_finished) \
                                    .view(-1, alive_seq.size(-1))
        
        # Reorder past_key_values
        select_indices = batch_index.view(-1)
        encoder_hidden_states = encoder_hidden_states.index_select(0, select_indices)
        past_key_values = recursive_apply(past_key_values, lambda x: x.index_select(0, select_indices))

    non_proper_batch = (~reach_num_return_sequences).nonzero().view(-1)
    non_proper_batch = [batch_offset[idx].item() for idx in non_proper_batch]
    if non_proper_batch:
        logger.warning("Batches {} do not have enough number of hypotheses, i.e. {}".format(
            ", ".join(non_proper_batch), ", ".join([len(hypotheses[idx]) for idx in non_proper_batch])
        ))
    for idx in range(batch_size):
        hypotheses[idx].sort(key=lambda x: x[0], reverse=True)
        hypotheses[idx] = hypotheses[idx][:num_return_sequences]
    logger.info("Maximum duplicated prefix: {}".format(maximum_num_per_beam))
    return hypotheses


def main():
    # TODO: write some test
    from transformers import (
        BartForConditionalGeneration, BartTokenizer, AutoTokenizer,
        PegasusTokenizer, PegasusForConditionalGeneration, T5ForConditionalGeneration
    )
    global logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="VietAI/vit5-base-vietnews-summarization")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base-vietnews-summarization")
    model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    model.to("cuda")
    model.eval()

    data = []
    with open(args.data_path, "r") as reader:
        for line in reader:
            data.append(json.loads(line.strip()))
    
    tokenized_data = []
    for item in tqdm(data):
        raw_texts = [doc['raw_text'] for doc in item['single_documents']]
        input_text = " ".join(raw_texts)
        input_ids = tokenizer(input_text, max_length=4096, truncation=True, return_tensors='pt').input_ids
        tokenized_data.append(input_ids)

    writer = open(args.output_path, "w")
    compact_data = []
    for idx, input_ids in tqdm(enumerate(tokenized_data), total=len(tokenized_data), desc="Sample"):
        with torch.no_grad():
            candidates = greedy_multiple(
                model,
                tokenizer,
                input_ids=input_ids.to("cuda"),
                beam_size=5,
                min_length=100,
                max_length=374,
                alpha=1.0,
                block_trigram=True,
                num_return_sequences=16,
                do_sample=False,
                max_candidates_per_beam=1
            )
        output_item = {**data[idx], 'candidates': candidates}
        compact_data.append(output_item)
        writer.write(json.dumps(output_item) + "\n")
    writer.close()


if __name__ == "__main__":
    main()
