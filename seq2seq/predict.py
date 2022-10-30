import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base-vietnews-summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base-vietnews-summarization")
# model = AutoModelForSeq2SeqLM.from_pretrained("/media/lvloi/projects/vlsp-2022/seq2seq/pretrained/checkpoints/checkpoint-231")
model.to("cuda")

with open("input.txt", "r") as reader:
    document = reader.read().strip()

document = tokenizer(document)
input_ids = torch.tensor([document.input_ids]).to("cuda")
print("Input length: {}".format(len(document.input_ids)))
min_len = int(len(document.input_ids) / (3.6813 + 1.5))
max_len = int(len(document.input_ids) / (3.6813 - 1.5))
output = model.generate(
    input_ids=input_ids,
    min_length=min_len,
    max_length=max_len,
    num_beams=5,
    do_sample=True
)
with tokenizer.as_target_tokenizer():
    output = tokenizer.decode(output[0], clean_up_tokenization_spaces=False, skip_special_tokens=True)

with open("output.txt", "w") as writer:
    writer.write(output)