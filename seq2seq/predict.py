import json
import time
import torch
import logging
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict_single(model, data, tokenizer, args):
    tokenized_data = []
    for item in data:
        raw_text = item['single_documents'][0]['raw_text']
        inputs = tokenizer(raw_text, return_tensors='pt')
        tokenized_data.append(inputs.input_ids)
    
    # generate prediction
    with open(args.output_path, "w") as writer:
        with torch.no_grad():
            for input_ids in tqdm(tokenized_data):
                candidates = model.generate(
                    input_ids=input_ids.to(model.device),
                    max_length=256,
                    num_beams=16,
                    length_penalty=1.0,
                    do_sample=True
                )
                with tokenizer.as_target_tokenizer():
                    outputs = [
                        tokenizer.decode(cand, clean_up_tokenization_spaces=False, skip_special_tokens=True)
                        for cand in candidates
                    ]
                for out in outputs:
                    writer.write(out + "\n")


def predict_all(model, data, tokenizer, args):
    logger.info("Tokenizing data...")
    t0 = time.perf_counter()
    tokenized_data = []
    doc_count = 0
    for item in tqdm(data):
        raw_texts = [doc['raw_text'] for doc in item['single_documents']]
        doc_count += len(raw_texts)
        tokenized_item = []
        for raw_text in raw_texts:
            inputs = tokenizer(raw_text, return_tensors='pt')
            tokenized_item.append(inputs.input_ids)
        tokenized_data.append(tokenized_item)
    
    progress_bar = tqdm(total=doc_count)
    output_data = []
    writer = open(args.output_path, "w")
    with torch.no_grad():
        for idx, item in enumerate(tokenized_data):
            output_candidates = []
            for input_ids in item:
                candidates = model.generate(
                    input_ids=input_ids.to(model.device),
                    max_length=256,
                    num_beams=16,
                    length_penalty=1.0,
                    do_sample=False
                )
                with tokenizer.as_target_tokenizer():
                    outputs = [
                        tokenizer.decode(cand, clean_up_tokenization_spaces=False, skip_special_tokens=True)
                        for cand in candidates
                    ]
                output_candidates.extend(outputs)
                progress_bar.update(1)
            raw_texts = [doc['raw_text'] for doc in data[idx]['single_documents']]
            output_item = {'src': raw_texts, 'candidates': output_candidates}
            writer.write(json.dumps(output_item) + "\n")
    
    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", required=True,
                        help="Path to the test file, in jsonlines format.")
    parser.add_argument("--model-path", required=True,
                        help="Path to the directory containing pytorch_model.bin file.")
    parser.add_argument("--tokenizer-path", default="VietAI/vit5-base-vietnews-summarization")
    parser.add_argument("--gpuid", default=0, type=int)
    parser.add_argument("--output-path", default="logs/candidate.out")
    parser.add_argument("--mode", default="single", choices=['single', 'all'])
    args = parser.parse_args()

    # load data
    data = []
    with open(args.test_file, "r") as reader:
        for line in reader:
            data.append(json.loads(line.strip()))
    
    # load model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    device = torch.device(f"cuda:{args.gpuid}")
    model.to(device)
    model.eval()

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    if args.mode == 'single':
        predict_single(model, data, tokenizer, args)
    else:
        predict_all(model=model, data=data, tokenizer=tokenizer, args=args)


if __name__ == "__main__":
    main()
