import csv
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments
from tqdm import tqdm
from torch.utils.data import DataLoader

model_name = "VietAI/vit5-base-vietnews-summarization" # or "VietAI/vit5-large-vietnews-summarization"
tokenizer = AutoTokenizer.from_pretrained(model_name)  
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.to('cuda:1')

from datasets import load_metric

metric = load_metric("rouge")

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["inputs"], max_length=1024, truncation=True, padding=True
    )

    raw_inputs = tokenizer(examples['inputs'])
    
    labels = tokenizer(
        examples["labels"], max_length=256, truncation=True, padding=True
    )
    model_inputs['labels'] = labels['input_ids']
    model_inputs['input_ids'] = model_inputs['input_ids']
    model_inputs['raw_input_ids'] = raw_inputs.input_ids
    return model_inputs

input_lines = []
label_lines = []
with open('valid_vlsp_2022.tsv') as file:
  csvreader = csv.reader(file, delimiter='\t')
  for line in csvreader:
    # line = line.strip().split('\t')
    
    # VietAI/vit5-large-vietnews-summarization need prefix vietnews: 
    if 'base' in model_name:
      input = line[0]
    else:
      input = "vietnews: " + line[0]

    input_lines.append(input)
    label_lines.append(line[1])



input_lines  = input_lines
label_lines = label_lines
dict_obj = {'inputs': input_lines, 'labels': label_lines}

dataset = Dataset.from_dict(dict_obj)
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=['inputs'], num_proc=10)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt")

import torch 
import numpy as np
metrics = load_metric('rouge')

# max_target_length = 256
dataloader = torch.utils.data.DataLoader(tokenized_datasets, collate_fn=data_collator, batch_size=1)

predictions = []
references = []
for i, batch in enumerate(tqdm(dataloader)):
  doc_len = len(batch['raw_input_ids'][0])
  min_len = int(doc_len / (3.6813 + 1.5))
  max_len = int(doc_len / (3.6813 - 1.5))
  outputs = model.generate(
      input_ids=batch['input_ids'].to('cuda:1'),
      max_length=max_len,
      min_length=min_len,
      attention_mask=batch['attention_mask'].to('cuda:1'),
      num_beams=5,
      do_sample=True
  )
  with tokenizer.as_target_tokenizer():
    outputs = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in outputs]

    labels = np.where(batch['labels'] != -100,  batch['labels'], tokenizer.pad_token_id)
    actuals = [tokenizer.decode(out, clean_up_tokenization_spaces=False, skip_special_tokens=True) for out in labels]
  predictions.extend(outputs)
  references.extend(actuals)
  metrics.add_batch(predictions=outputs, references=actuals)

with open("logs/vlsp-2022/candidate.out", "w") as writer:
  writer.write("\n".join(predictions))
with open("logs/vlsp-2022/gold.out", "w") as writer:
  writer.write("\n".join(references))
result = metrics.compute()
print(result)