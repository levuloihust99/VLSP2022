import json
import argparse

from sumy.parsers.plaintext import PlaintextParser #We're choosing a plaintext parser here, other parsers available for HTML etc.
from sumy.nlp.tokenizers import Tokenizer 
from sumy.summarizers.lex_rank import LexRankSummarizer #We're choosing Lexrank, other algorithms are also built in


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", required=True,
                        help="Path to the input data file.")
    parser.add_argument("--output-path", required=True,
                        help="Path to the output file that contains summaries.")
    args = parser.parse_args()

    data = []
    with open(args.file_path, "r") as reader:
        for line in reader:
            data.append(json.loads(line.strip()))

    summarizer = LexRankSummarizer()
    
    summaries = []
    for item in data:
        input_text = " ".join([doc['raw_text'] for doc in item['single_documents']])
        parser = PlaintextParser.from_string(input_text, Tokenizer("english"))
        summary = summarizer(parser.document, 5)
        summaries.append(" ".join([str(s) for s in summary]))

    with open(args.output_path, "w") as writer:
        for summary in summaries:
            writer.write(summary + "\n")


if __name__ == "__main__":
    main()
