import os
import json
import time
import torch
import logging

from torch.nn import CrossEntropyLoss
from .grad_cache import RandContext

logger = logging.getLogger(__name__)


class BertExtractiveTrainer(object):
    def __init__(
        self,
        model,
        cfg,
        optimizer,
        scheduler,
        dataloader,
        device,
        saved_state
    ):
        self.model = model
        self.cfg = cfg
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.device = device
        self.saved_state = saved_state
        self.data_iterator = iter(dataloader)
        self.loss_fn = CrossEntropyLoss(reduction='sum')
        self._marked_time = time.perf_counter()
    
    def save_checkpoint(self, step):
        cp_name = "bert_ext.{}.pt".format(step + 1)
        model_state_dict = self.model.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        scheduler_state_dict = self.scheduler.state_dict()
        training_args = self.cfg._content

        cp_dir = os.path.join(self.cfg.output_dir, cp_name)
        if not os.path.exists(cp_dir):
            os.makedirs(cp_dir)
        save_state = {
            'model_dict': model_state_dict,
            'optimizer_dict': optimizer_state_dict,
            'scheduler_dict': scheduler_state_dict,
            'step': step + 1,
        }
        torch.save(save_state, os.path.join(cp_dir, 'checkpoint.pt'))
        with open(os.path.join(cp_dir, 'training_args.json'), "w") as writer:
            json.dump(training_args, writer, indent=4, ensure_ascii=False)
        logger.info("Checkpoint saved at {}".format(cp_dir))

    def train(self):
        logger.info("Start training...")
        logger.info("Total updates: {}".format(self.cfg.total_updates))
        logger.info("Model has trained for {} steps".format(self.saved_state['trained_steps']))
        self.model.train()
        batches = []
        self._mark_time = time.perf_counter()
        for step in range(self.saved_state['trained_steps'], self.cfg.total_updates):
            batches = self.fetch_batches()
            per_update_loss = self.train_step(batches)
            if (step + 1) % self.cfg.logging_steps == 0:
                logger.info("Step {}/{}  |  Loss: {} | Time elapsed: {}".
                    format(step + 1, self.cfg.total_updates, per_update_loss, time.perf_counter() - self._marked_time))
                self._mark_time = time.perf_counter()
            if (step + 1) % self.cfg.save_checkpoint_step == 0:
                self.save_checkpoint(step)
    
    def fetch_batches(self):
        batches = []
        for _ in range(self.cfg.gradient_accumulate_steps):
            try:
                batch = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.dataloader)
                batch = next(self.data_iterator)
            batches.append(batch)
        return batches

    def train_step(self, batches):
        normalization = sum([batch['labels'].size(0) for batch in batches])
        per_update_loss = 0.0
        for batch in batches:
            # DEBUG
            batch['num_sents'] = batch['num_sents'][:1]
            truncate_len = torch.sum(batch['num_sents'])
            batch['sentence/input_ids'] = batch['sentence/input_ids'][:truncate_len]
            batch['sentence/attn_mask'] = batch['sentence/attn_mask'][:truncate_len]
            batch['labels'] = batch['labels'][:truncate_len]
            per_batch_loss_unnorm = self._computation_grad_cache(
                    batch, normalization)
            print("Grad cache loss: {}".format(per_batch_loss_unnorm.item()))
            per_batch_loss_unnorm = self._computation_no_cache(
                    batch, normalization)
            print("No cache loss: {}".format(per_batch_loss_unnorm.item()))
            raise KeyboardInterrupt
            if self.cfg.grad_cache:
                per_batch_loss_unnorm = self._computation_grad_cache(
                    batch, normalization)
            else:
                per_batch_loss_unnorm = self._computation_no_cache(
                    batch, normalization)
            per_update_loss += per_batch_loss_unnorm
        per_update_loss /= normalization

        self.optimizer.step()
        self.scheduler.step()
        self.model.zero_grad()

        return per_update_loss

    def _computation_no_cache(self, batch, normalization):
        sentence_input_ids = batch['sentence/input_ids'].to(self.device)
        sentence_attn_mask = batch['sentence/attn_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        num_sents = batch['num_sents'].to(self.device)

        encoder_outputs = self.encoder(
            input_ids=sentence_input_ids, attention_mask=sentence_attn_mask, return_dict=True)
        sequence_output = encoder_outputs.last_hidden_state
        sentence_embs = sequence_output[:, 0, :]

        max_num_sent = torch.max(num_sents)
        batch_sentence_embs = []
        batch_inter_attn_mask = []
        count = 0
        for num_sent in num_sents:
            sent_emb = sentence_embs[count: count + num_sent]
            count += num_sent
            inter_attn_mask = [1] * num_sent
            num_padding = max_num_sent - num_sent
            if num_padding > 0:
                sent_emb = torch.cat(
                    [sent_emb, torch.zeros(num_padding, sent_emb.size(1))],
                    dim=0
                )
                inter_attn_mask += [0] * num_padding
            inter_attn_mask = torch.tensor(inter_attn_mask)
            batch_sentence_embs.append(sent_emb)
            batch_inter_attn_mask.append(inter_attn_mask)

        batch_sentence_embs = torch.stack(batch_sentence_embs, dim=0)
        batch_inter_attn_mask = torch.stack(batch_inter_attn_mask, dim=0)

        inter_outputs = self.inter_encoder(
            hidden_states=batch_sentence_embs, attention_mask=batch_inter_attn_mask, return_dict=True)
        inter_sent_embs = inter_outputs.last_hidden_state
        active_mask = batch_inter_attn_mask.view(-1)
        inter_sent_embs = inter_sent_embs.view(-1,
                                               inter_sent_embs.size(2))[active_mask]

        inter_sent_logits = self.cls(inter_sent_embs)
        loss = self.loss_fn(inter_sent_logits, labels) / normalization
        loss.backward()
        return loss.item()

    def _computation_grad_cache(self, batch, normalization):
        chunk_size = self.cfg.chunk_size
        # [B, seq_len]
        sentence_input_ids = batch['sentence/input_ids'].to(self.device)
        # [B, seq_len]
        sentence_attn_mask = batch['sentence/attn_mask'].to(self.device)
        labels = batch['labels'].to(self.device)  # [B]
        num_sents = batch['num_sents'].to(
            self.device)  # [batch_size]: batch_size < B

        bsz = sentence_input_ids.size(0)
        num_chunks = (bsz - 1) // chunk_size + 1
        base_size = bsz // num_chunks
        chunk_sizes = [base_size] * num_chunks
        remainder = bsz % num_chunks
        remainder_arr = [1] * remainder + [0] * (num_chunks - remainder)
        chunk_sizes = [size + added for size,
                       added in zip(chunk_sizes, remainder_arr)]

        # chunking input
        chunked_sentence_input_ids = []
        chunked_sentence_attn_mask = []
        idx = 0
        for chunk_size in chunk_sizes:
            chunked_sentence_input_ids.append(
                sentence_input_ids[idx: idx + chunk_size])
            chunked_sentence_attn_mask.append(
                sentence_attn_mask[idx: idx + chunk_size])
            idx += chunk_size

        # chunking forward
        rnds = []
        all_sentence_embeddings = []
        for id_chunk, attn_chunk in zip(chunked_sentence_input_ids, chunked_sentence_attn_mask):
            rnds.append(RandContext(id_chunk, attn_chunk))
            # get sentence embeddings
            with torch.no_grad():
                outputs = self.model.encoder(
                    input_ids=id_chunk, attention_mask=attn_chunk, return_dict=True)
                chunked_sequence_output = outputs.last_hidden_state
                chunked_pooled_output = chunked_sequence_output[:, 0, :]
                all_sentence_embeddings.append(chunked_pooled_output)
        all_sentence_embeddings = torch.cat(all_sentence_embeddings, dim=0)

        # loss calculation
        all_sentence_embeddings = all_sentence_embeddings.requires_grad_()
        max_num_sent = torch.max(num_sents)
        batch_sentence_embs = []
        batch_inter_attn_mask = []
        count = 0
        for num_sent in num_sents:
            sent_emb = all_sentence_embeddings[count: count + num_sent]
            count += num_sent
            inter_attn_mask = [1] * num_sent
            num_padding = max_num_sent - num_sent
            if num_padding > 0:
                sent_emb = torch.cat(
                    [sent_emb, torch.zeros(num_padding, sent_emb.size(1)).to(self.device)],
                    dim=0
                )
                inter_attn_mask += [0] * num_padding
            inter_attn_mask = torch.tensor(inter_attn_mask).to(self.device)
            batch_sentence_embs.append(sent_emb)
            batch_inter_attn_mask.append(inter_attn_mask)

        batch_sentence_embs = torch.stack(batch_sentence_embs, dim=0)
        batch_inter_attn_mask = torch.stack(batch_inter_attn_mask, dim=0)
        extended_batch_inter_attn_mask = batch_inter_attn_mask[:, None, None, :]
        extended_batch_inter_attn_mask = (1.0 - extended_batch_inter_attn_mask) * -10000.0
        inter_outputs = self.model.inter_encoder(
            hidden_states=batch_sentence_embs, attention_mask=extended_batch_inter_attn_mask, return_dict=True)
        inter_sent_embs = inter_outputs.last_hidden_state
        active_mask = batch_inter_attn_mask.view(-1).to(torch.bool)
        inter_sent_embs = inter_sent_embs.view(-1,
                                               inter_sent_embs.size(2))[active_mask]

        inter_sent_logits = self.model.cls(inter_sent_embs)
        loss = self.loss_fn(inter_sent_logits, labels)
        _loss = loss / normalization
        _loss.backward()

        # chunking forward backward
        grads = all_sentence_embeddings.grad  # [B, hidden_size]
        # chunking grads
        chunked_grads = []
        idx = 0
        for chunk_size in chunk_sizes:
            chunked_grads.append(grads[idx: idx + chunk_size])
            idx += chunk_size

        # forward backward
        for id_chunk, attn_chunk, grad_chunk, rnd in zip(
            chunked_sentence_input_ids, chunked_sentence_attn_mask, chunked_grads, rnds
        ):
            with rnd:
                outputs = self.model.encoder(
                    input_ids=id_chunk, attention_mask=attn_chunk, return_dict=True)
                chunked_sequence_output = outputs.last_hidden_state
                chunked_pooled_output = chunked_sequence_output[:, 0, :]
                surrogate = torch.dot(
                    chunked_pooled_output.flatten(), grad_chunk.flatten())

            surrogate.backward()

        return loss.item()
