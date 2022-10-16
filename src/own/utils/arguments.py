import argparse


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--data-path")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--max-encoder-sequence-length", type=int)
    parser.add_argument("--max-decoder-sequence-length", type=int)
    parser.add_argument("--save-checkpoint-steps", type=int)
    parser.add_argument("--decoder-num-hidden-layers", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--valid-batch-size", type=int)
    parser.add_argument("--dev-data-path")
    parser.add_argument("--perform-validation", type=eval)
    parser.add_argument("--keep-checkpoint-max", type=int)
    parser.add_argument("--logging-steps", type=int)
    parser.add_argument("--encoder-pretrained-path")
    parser.add_argument("--decoder-pretrained-path")
    parser.add_argument("--encoder-learning-rate", type=float)
    parser.add_argument("--decoder-learning-rate", type=float)
    parser.add_argument("--use-encoder-embs", type=eval)
    parser.add_argument("--max-grad-norm", type=float)
    parser.add_argument("--gradient-accumulate-steps", type=int)
    parser.add_argument("--label-smoothing", type=float)
