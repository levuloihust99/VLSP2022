import argparse
from rouge_metric import PyRouge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-path", required=True)
    parser.add_argument("--reference-path", required=True)
    args = parser.parse_args()

    hypotheses = []
    with open(args.candidate_path, "r") as reader:
        for line in reader:
            hypotheses.append(line.strip())
    
    references = []
    with open(args.reference_path, "r") as reader:
        for line in reader:
            references.append([line.strip()])

    # Evaluate document-wise ROUGE scores
    rouge = PyRouge(rouge_n=(1, 2, 4), rouge_l=True, rouge_w=True,
                    rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4)
    scores = rouge.evaluate(hypotheses, references)
    print(scores)


if __name__ == "__main__":
    main()
