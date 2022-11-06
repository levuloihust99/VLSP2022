import os
import argparse

import torch
from transformers import AutoModelForSeq2SeqLM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained-model-path", default="VietAI/vit5-base-vietnews-summarization")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model_path)
    saved_state = torch.load(args.checkpoint_path, map_location=lambda s, t: s)
    saved_state = {k[len('model.'):]: v for k, v in saved_state.items()}
    model.load_state_dict(saved_state)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
