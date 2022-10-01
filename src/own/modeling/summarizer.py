import torch
from torch import nn
from typing import Optional, Tuple

from .modeling_utils import tile, recursive_apply


class AbsSummarizer(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        tokenizer,
        decoder_start_token_id: int = 1,
        decoder_end_token_id: int = 2,
        alpha: float = 0.95,
        block_trigram: bool = True,
    ):
        super(AbsSummarizer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.decoder_start_token_id = decoder_start_token_id
        self.decoder_end_token_id = decoder_end_token_id
        self.alpha = alpha
        self.block_trgram = block_trigram
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[Tuple[torch.Tensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
    ):
        if encoder_hidden_states is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                return_dict=True
            )
            encoder_hidden_states = encoder_outputs.last_hidden_state
        
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask
        )
        return decoder_outputs.logits

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        beam_size: int = 1,
        min_length: int = 0,
        max_length: int = 100,
    ):
        if input_ids is None:
            assert inputs_embeds is not None, \
                "At least 'input_ids' or 'inputs_embeds' must be provided."
            batch_size = inputs_embeds.size(0)
        else:
            batch_size = input_ids.size(0)

        if encoder_hidden_states is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
                return_dict=True
            )
            encoder_hidden_states = encoder_outputs.last_hidden_state
        encoder_hidden_states = tile(encoder_hidden_states, beam_size, dim=0)

        # Structure that holds finished hypotheses
        hypotheses = [[] for _ in range(batch_size)]

        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=self.device)
        beam_offset = torch.arange(
            0,
            batch_size * beam_size,
            step=beam_size,
            dtype=torch.long,
            device=self.device)
        alive_seq = torch.full(
            [batch_size * beam_size, 1],
            self.decoder_start_token_id,
            dtype=torch.long,
            device=self.device
        )

        # Give full probability to the first beam on the first step.
        topk_log_probs = torch.tensor([0.0] + [float("-inf")] * (beam_size - 1)).repeat(batch_size) # [batch_size * beam_size]

        past_key_values = None
        results = {
            "scores": [None for _ in range(batch_size)],
            "predictions": [None for _ in range(batch_size)]
        }
        for step in range(max_length):
            decoder_input_ids = alive_seq[:, -1:] # [batch_size * beam_size, 1]
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                use_cache=True,
                past_key_values=past_key_values
            )
            past_key_values = decoder_outputs.past_key_values
            logits = decoder_outputs.logits # [batch_size * beam_size, 1, vocab_size]
            vocab_size = logits.size(-1)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1).squeeze(1) # [batch_size * beam_size, vocab_size]

            if step < min_length:
                log_probs[:, self.decoder_end_token_id] = -1e20
            
            # sum of log probs of the current sequence
            log_probs += topk_log_probs.view(-1).unsqueeze(1) # [batch_size * beam_size, vocab_size]

            length_penalty = ((5.0 + (step + 1)) / 6.0) ** self.alpha
            cur_scores = log_probs / length_penalty # [batch_size * beam_size, vocab_size]

            # if trigram is repeated in a beam, ignore that beam (assign large minus score)
            if self.block_trigram:
                cur_length = alive_seq.size(1)
                if cur_length > 3:
                    for idx in range(alive_seq.size(0)):
                        fail = False
                        words = [int(w) for w in alive_seq[idx]]
                        words = self.tokenizer.decode(words).split()
                        if len(words) <=3:
                            continue
                        trigrams = [(words[i - 1], words[i], words[i + 1]) for i in range(1, len(words) - 1)]
                        trigram = tuple(trigrams[-1])
                        if trigram in trigrams[:-1]:
                            fail = True
                        if fail:
                            cur_scores[idx] = -1e20

            cur_scores = cur_scores.reshape(-1, beam_size * vocab_size) # [batch_size, beam_size * vocab_size]
            topk_scores, topk_ids = cur_scores.topk(beam_size, dim=-1)
            
            # Recover log probs
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids
            topk_beam_index = topk_ids.div(vocab_size, rounding_mode="trunc")
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation
            batch_index = (
                topk_beam_index
                + beam_offset.unqueeze(1)
            ) # [batch_size, beam_size]
            select_indices = batch_index.view(-1) # [batch_size * beam_size]

            # Append last prediction
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                topk_ids.view(-1, 1)],
                dim=-1
            )

            is_finished = topk_ids.eq(self.decoder_end_token_id) # [batch_size, beam_size]
            if step + 1 == max_length:
                is_finished.fill_(1)
            # End condition is top beam is finished
            end_condition = is_finished[:, 0].eq(1) # [batch_size]
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
                    # If the batch reached the end, save the n_best hypotheses
                    if end_condition[batch_idx]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True
                        )
                        score, pred = best_hyp[0]

                        results["scores"][batch_idx] = score
                        results["predictions"][batch_idx] = pred
                
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
            past_key_values = recursive_apply(past_key_values, lambda x: x.index_select(select_indices))

        return results
