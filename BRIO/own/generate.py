import torch
from torch import nn
from typing import Optional, Tuple
import logging

from .utils import tile, recursive_apply


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
    num_sampling: int = 32
):
    """Decoding process stops either when the output sequence length exceeds `max_length` or `num_return_sequences` hypotheses is reached."""

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
    results = {
        "scores": [None for _ in range(batch_size)],
        "predictions": [None for _ in range(batch_size)]
    }
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
        # ignore beams that ended
        for idx in range(alive_seq.size(0)):
            prev_generated_token_id = alive_seq[idx][-1]
            if prev_generated_token_id == decoder_end_token_id:
                cur_scores[idx] = -1e20 # ignore beam `idx`

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
            topk_scores, topk_ids = cur_scores.topk(beam_size, dim=-1)
        
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
        hypotheses[idx] = hypotheses[idx][:beam_size]
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
    tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base-vietnews-summarization")
    model = T5ForConditionalGeneration.from_pretrained("VietAI/vit5-base-vietnews-summarization")
    model.to("cuda")
    model.eval()
    inputs = tokenizer('Sáng 26/10, Quốc hội thảo luận tại tổ về dự thảo Nghị quyết thí điểm cấp quyền lựa chọn sử dụng biển số ô tô thông qua đấu giá. Theo dự thảo được Bộ trưởng Công an thông tin trước đó tại Quốc hội, người được trúng đấu giá sẽ được giữ lại biển số trúng đấu giá khi chuyển nhượng, cho tặng, thừa kế xe ô tô để đăng ký cho xe khác thuộc sở hữu của mình. Trong thời hạn 12 tháng kể từ ngày được cấp văn bản xác nhận trúng đấu giá, người trúng đấu giá phải thực hiện thủ tục đăng ký xe tại cơ quan đăng ký xe để gắn biển số với xe. Nếu quá hạn mà không đăng ký biển số đó gắn với xe thì cơ quan có thẩm quyền sẽ thu hồi biển số trúng đấu giá. Đồng tình với nội dung dự thảo nghị quyết, Chủ tịch UBND TP Hà Nội Trần Sỹ Thanh cho rằng quy định đó sẽ tránh được chuyện "đầu cơ" biển số, mua bán biển số xe gây phức tạp. Về giá khởi điểm của biển số được đưa ra đấu giá (vùng 1 gồm Hà Nội, TPHCM là 40 triệu đồng và vùng 2 gồm các địa phương còn lại là 20 triệu đồng/biển số) mà dự thảo đưa ra, ông Trần Sỹ Thanh lo ngại "sẽ loạn" Ông đề nghị chỉ quy định mức sàn của giá khởi điểm, còn lại giao HĐND các tỉnh, thành quyết định mức giá khởi điểm, bước giá. "Nên giao HĐND quyết, đừng chê tỉnh nghèo. Nhà nước có thể nghèo chứ dân không nghèo, ví dụ Đắk Lắk, xe xịn còn nhiều hơn Đà Nẵng"- Chủ tịch UBND TP Hà Nội Trần Sỹ Thanh nói. Ông phân tích, ở TP Hà Nội bước giá phải 20, 40, 50 triệu đồng thì đấu giá mới nhanh, có khi chỉ 10 phút xong. Tiền thu được từ đấu giá biển số xe nên được đưa về ngân sách địa phương. Thậm chí, Chủ tịch Hà Nội Trần Sỹ Thanh còn cho rằng việc thí điểm đấu giá biển số không được phá vỡ nguyên tắc quản lý xe theo địa giới hành chính. Đơn cử như nếu đấu giá tập trung trên phạm vi cả nước thì có thể xảy ra chuyện toàn bộ người dân phía Bắc sẽ đấu giá biển số xe Hà Nội, gây khó khăn cho quản lý. Vợ chồng bỏ 5 tỷ đấu giá biển số, khi ly hôn giải quyết thế nào? Ở tổ TPHCM, đại biểu Quốc hội Trương Trọng Nghĩa cho rằng nội dung dự thảo nghị quyết đã tạo ra một thị trường, biển số xe từ chỗ không là tài sản công, phục vụ công tác quản lý, thì bây giờ có thể có giá lên tới vài tỉ đồng, thậm chí còn cao hơn. Do đó phải có quy định chặt chẽ để quản lý, bởi tài sản còn liên quan đến vấn đề chuyển nhượng, thừa kế, cho tặng. Đáng chú ý, ông Nghĩa nhận định, việc đấu giá quyền sử dụng biển số ô tô còn liên quan đến tài sản vợ chồng và rất nhiều vấn đề khác. "Vợ chồng bỏ ra 5 tỷ đồng để đấu giá lấy một biển số ôtô, thì khi ly hôn chia tài sản như thế nào"- ông Nghĩa đặt vấn đề. Ông Nghĩa đề nghị cơ quan thẩm tra, cơ quan soạn thảo thận trọng, nếu các nội dung trong dự thảo chưa ổn thì không vội vàng, phải yêu cầu xây dựng quy định chặt chẽ. Trong khi đó, Phó Chủ nhiệm Ủy ban Pháp luật Nguyễn Phương Thủy băn khoăn khi dự thảo quy định theo hướng người được trúng đấu giá được giữ lại biển số trúng đấu giá khi chuyển nhượng, cho tặng, thừa kế xe để đăng ký cho xe khác thuộc sở hữu của mình. "Tôi thấy thực sự rất băn khoăn vì mục tiêu của biển số xe là để quản lý phương tiện mà bây giờ lại tách rời với phương tiện để như một tài sản có thể chuyển nhượng được, có thể chuyển từ xe nọ sang xe kia. Việc này sẽ rất phức tạp trong quản lý, nhất là khi chúng ta đang thực hiện thí điểm vấn đề này"- bà Thủy phân tích. Từ đó vị đại biểu đề nghị, biển số xe trúng đấu giá vẫn gắn với phương tiện, ngay khi mua bán, chuyển nhượng, thừa kế phương tiện đó. Khi nào hết vòng đời phương tiện thì biển số xe được thu hồi để đưa vào kho số đấu giá tiếp. "Biển số xe gắn với người có thể dẫn đến một số trường hợp chúng ta chưa lý giải được và dẫn đến một khả năng đầu cơ rất lớn. Người ta có thể đấu giá rất nhiều biển số để gắn cho xe giá rẻ, khi ai đó có nhu cầu mua biển cho xe sang, xe xịn thì sẽ bán lại"- Phó Chủ nhiệm Ủy ban Pháp luật nêu ý kiến.',
                        return_tensors='pt')

    # with torch.no_grad():
        # candidates = generate(
        #     model,
        #     tokenizer,
        #     input_ids=inputs.input_ids.to("cuda"),
        #     beam_size=16,
        #     min_length=100,
        #     max_length=256,
        #     alpha=1.0,
        #     block_trigram=True,
        #     num_return_sequences=16,
        #     do_sample=False
        # )
    # candidates = candidates[0]
    # texts = []
    # for cand in candidates:
    #     with tokenizer.as_target_tokenizer():
    #         texts.append(tokenizer.decode(cand[1], clean_up_tokenization_spaces=False, skip_special_tokens=True))


if __name__ == "__main__":
    main()