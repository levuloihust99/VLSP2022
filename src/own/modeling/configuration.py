import os
import json
import logging

logger = logging.getLogger(__name__)


class TrainingConfig(object):
    # params that must not be overriden when resuming training
    must_kept_params = [
        "seed",
        "data_path",
        "dev_data_path",
        "tokenizer_type",
        "tokenizer_path",
        "encoder_pretrained_path"
        "decoder_pretrained_path",
        "encoder_architecture",
        "decoder_architecture",
        "alpha",
        "block_trigram",
        "use_encoder_embs",
        "max_encoder_sequence_length",
        "max_decoder_sequence_length",
        "decoder_num_hidden_layers",
        "decoder_start_token_id",
        "decoder_end_token_id",
        "decoder_sep_token_id",
        "decoder_pad_token_id",
        "use_segmentation",
        "batch_size",
        "num_train_epochs",
        "weight_decay",
        "adam_epsilon",
        "beta1",
        "beta2",
        "max_grad_norm",
        "checkpoint_path",
        "keep_checkpoint_max"
    ]

    def __init__(self, **kwargs):
        self.seed = 42

        self.data_path = None
        self.dev_data_path = None
        self.tokenizer_type = "bert"
        self.tokenizer_path = "bert-base-uncased"
        self.encoder_pretrained_path = "bert-base-uncased"
        self.decoder_pretrained_path = None
        self.encoder_architecture = "bert"
        self.decoder_architecture = "bert"
        self.alpha = 0.95
        self.block_trigram = True
        self.use_encoder_embs = True

        self.max_encoder_sequence_length = 512
        self.max_decoder_sequence_length = 512
        self.decoder_num_hidden_layers = 6
        self.decoder_start_token_id = 1
        self.decoder_end_token_id = 2
        self.decoder_sep_token_id = 3
        self.decoder_pad_token_id = 0
        self.use_segmentation = True
        
        self.batch_size = 4
        self.valid_batch_size = 16
        self.perform_validation = True
        self.gradient_accumulate_steps = 1
        self.save_checkpoint_steps = 1000
        self.logging_steps = 50
        self.num_train_epochs = 50
        self.num_warmup_steps = 1000
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.max_grad_norm = 0.0
        self.encoder_learning_rate = 2e-5
        self.decoder_learning_rate = 2e-5
        self.inter_encoder_learning_rate = 2e-5
        self.label_smoothing = 0.1

        self.checkpoint_path = None
        self.keep_checkpoint_max = 5
        self.nb_gpu = None

        self.override_defaults(**kwargs)
    
    def override_defaults(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.__dict__:
                logger.warn("Unknown hparam " + k)
            self.__dict__[k] = v
            if k in {'data_path', 'checkpoint_path', 'dev_data_path'}:
                self.__dict__[k] = os.path.abspath(v)
    
    def override(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.__dict__:
                logger.warning("Unknown param: '{}'".format(k))
            old_value = self.__dict__[k]
            if old_value != v and k in self.must_kept_params:
                self.__dict__[k] = v
                logger.info("Override param '{}': {}  \u2b62  {}".format(k, old_value, v))

    def to_json(self):
        return json.loads(json.dumps(self.__dict__))
    
    def format(self):
        attributes = self.__dict__.keys()
        max_attr_name_len = max([len(attr) for attr in attributes])
        spaces = " " * 20
        arrow = "\u2b62"
        formatted_str = "\n"
        for attr_name, attr_value in self.__dict__.items():
            padded_spaces = " " * (max_attr_name_len - len(attr_name))
            line = attr_name + padded_spaces + spaces + arrow + " " * 5 + str(attr_value) + "\n"
            formatted_str += line
        return formatted_str
