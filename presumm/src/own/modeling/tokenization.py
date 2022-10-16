from transformers import BertTokenizer


def init_tokenizer(tokenizer_type, tokenizer_path):
    if tokenizer_type == 'bert':
        return BertTokenizer.from_pretrained(tokenizer_path)
    else:
        raise Exception("Tokenizer type '{}' is not supported.".format(tokenizer_type))