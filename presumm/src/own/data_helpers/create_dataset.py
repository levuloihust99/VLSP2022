"""Create a ByteDataset from .pt files released in `https://github.com/nlpyang/PreSumm`"""

import os
import re
import argparse
import logging
import pickle
from typing import Text
from pathlib import Path
from tqdm import tqdm
import torch

from ..utils.logging_utils import add_color_formater
from .datasets import ByteDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
add_color_formater(logging.root)


def create_dataset(files, portion_dir):
    idxs_writer = open(os.path.join(portion_dir, "idxs.pkl"), "wb")
    dataset_size_place_holder = (0).to_bytes(4, 'big')
    idxs_writer.write(dataset_size_place_holder)

    data_writer = open(os.path.join(portion_dir, "data.pkl"), "wb")

    num_samples = 0
    for f in files:
        logger.info("Writing samples in {}...".format(f))
        chunk = torch.load(f)
        for sample in tqdm(chunk):
            idxs_writer.write(data_writer.tell().to_bytes(6, 'big'))
            pickle.dump(sample, data_writer)
            num_samples += 1
    
    data_writer.close()
    idxs_writer.seek(0, 0)
    idxs_writer.write(num_samples.to_bytes(4, 'big'))
    idxs_writer.close()


def process(
    input_path: Text,
    output_path: Text,
    prefix: Text
):
    logger.info("Load presumm data...")
    train_prefix = f"{prefix}.train"
    valid_prefix = f"{prefix}.valid"
    test_prefix  = f"{prefix}.test"
    
    all_files = os.listdir(input_path)
    train_files = []
    valid_files = []
    test_files  = []
    for f in all_files:
        if f.startswith(train_prefix):
            train_files.append(os.path.join(input_path, f))
        elif f.startswith(valid_prefix):
            valid_files.append(os.path.join(input_path, f))
        elif f.startswith(test_prefix):
            test_files.append(os.path.join(input_path, f))
    
    for portion, files in zip(["train", "valid", "test"], [train_files, valid_files, test_files]):
        logger.info(f"************* {portion.capitalize()} files *************")
        portion_dir = os.path.join(output_path, portion)
        Path(portion_dir).mkdir(exist_ok=True)
        files = sorted(files, key=lambda x: int(re.search(r"\d+", x).group(0)))
        create_dataset(files, portion_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--presumm-data-path", default="bert_data_cnndm_final")
    parser.add_argument("--prefix", default="cnndm")
    parser.add_argument("--output-path", default="data")
    args = parser.parse_args()

    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    process(args.presumm_data_path, output_path, args.prefix)


if __name__ == "__main__":
    main()
