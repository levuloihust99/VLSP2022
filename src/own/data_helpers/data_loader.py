import argparse
from typing import Text
from .datasets import ByteDataset


def create_dataloader(
    data_path: Text,
    training: bool = True
):
    dataset = ByteDataset(data_path, idx_record_size=6)
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data/train")
    args = parser.parse_args()
    dataset = create_dataloader(data_path=args.data_path, training=False)
    print("done")


if __name__ == "__main__":
    main()
