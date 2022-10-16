import argparse


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--data-path")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--max-encoder-sequence-length", type=int)
    parser.add_argument("--max-decoder-sequence-length", type=int)
    parser.add_argument("--save-checkpoint-steps", type=int)
