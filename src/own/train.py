import os
import json
import glob
import copy
import torch
import logging
import signal
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .modeling.optimization import AbsSummarizerOptimizer, create_optimizers_and_schedulers
from .modeling.summarizer import AbsSummarizer
from .modeling.trainer import SummarizerTrainer
from .modeling.configuration import TrainingConfig
from .data_helpers.data_loader import create_dataloader
from .modeling.model_builder.encoder import init_encoder
from .modeling.model_builder.decoder import init_decoder
from .modeling.tokenization import init_tokenizer
from .utils.arguments import add_arguments
from .utils.logging_utils import add_color_formatter


def override_defaults(hparams, args):
    for key in args:
        hparams[key] = args[key]
    return hparams


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def main():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29506'
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("--hparams", default="{}")
    add_arguments(parser)
    args = parser.parse_args()

    # < setup logging
    logging.basicConfig(level=logging.INFO)
    add_color_formatter(logging.root)
    # setup logging />

    # < config
    if args.hparams.endswith(".json"):
        with open(args.hparams, "r") as reader:
            hparams = json.load(reader)
    else:
        hparams = json.loads(args.hparams)
    args_json = copy.deepcopy(args.__dict__)
    args_json.pop('hparams')
    hparams = override_defaults(hparams, args_json)

    config = TrainingConfig(**hparams)
    nb_gpu = torch.cuda.device_count()
    assert nb_gpu == 0 or config.batch_size % nb_gpu == 0, \
        "Training with multi GPUs requires 'batch_size' to be divided by number of GPUs"
    config.nb_gpu = nb_gpu
    try:
        saved_state = torch.load(config.checkpoint_path, map_location=lambda s, t: s)
    except Exception:
        saved_state = None
    if saved_state:
        used_params = saved_state['params']
        config.override(**used_params)
    assert config.save_checkpoint_steps % config.gradient_accumulate_steps == 0, \
        "'save_checkpoint_steps' must be divided by 'gradient_accumulate_steps' for perfectly resuming training. \
        Got save_checkpoint_steps={} and gradient_accumulate_steps={}".format(
            config.save_checkpoint_steps, config.gradient_accumulate_steps)
    logging.info(config.format())
    # config />

    if config.nb_gpu > 1:
        mp = torch.multiprocessing.get_context('spawn')
        error_queue = mp.SimpleQueue()
        error_handler = ErrorHandler(error_queue=error_queue)
        workers = []
        for rank in range(config.nb_gpu):
            worker = mp.Process(target=run, args=(rank, config.nb_gpu, config, error_queue), daemon=True)
            workers.append(worker)
            worker.start()
            error_handler.add_child(worker.pid)
        
        for worker in workers:
            worker.join()
    else:
        rank = -1 if config.nb_gpu == 0 else 0
        setup_and_train(config=config, gpu_rank=rank, nb_gpu=config.nb_gpu)


def setup_and_train(config: TrainingConfig, gpu_rank: int, nb_gpu: int):
    # < logging
    logger = logging.getLogger(__name__)
    # logging />

    # < tokenizer
    tokenizer = init_tokenizer(tokenizer_type=config.tokenizer_type,
                                tokenizer_path=config.tokenizer_path)
    # tokenizer />

    # < data loader
    data_loader = create_dataloader(
        data_path=config.data_path,
        batch_size=config.batch_size,
        max_encoder_sequence_length=config.max_encoder_sequence_length,
        encoder_sep_token_id=tokenizer.sep_token_id,
        encoder_pad_token_id=tokenizer.pad_token_id,
        max_decoder_sequence_length=config.max_decoder_sequence_length,
        decoder_end_token_id=config.decoder_end_token_id,
        decoder_pad_token_id=config.decoder_pad_token_id,
        use_segmentation=config.use_segmentation,
        training=True
    )
    dataset_size = len(data_loader.dataset)
    steps_per_epoch = (dataset_size - 1) // config.batch_size + 1
    total_steps = steps_per_epoch * config.num_train_epochs
    # data loader />

    # < dev data loader
    if config.perform_validation:
        dev_data_loader = create_dataloader(
            data_path=config.dev_data_path,
            batch_size=config.valid_batch_size,
            max_encoder_sequence_length=config.max_encoder_sequence_length,
            encoder_sep_token_id=tokenizer.sep_token_id,
            encoder_pad_token_id=tokenizer.pad_token_id,
            max_decoder_sequence_length=config.max_decoder_sequence_length,
            decoder_end_token_id=config.decoder_end_token_id,
            decoder_pad_token_id=config.decoder_pad_token_id,
            use_segmentation=config.use_segmentation,
            training=False
        )
    else:
        dev_data_loader = None
    # dev data loader />

    # < model initialization
    encoder = init_encoder(
        architecture=config.encoder_architecture,
        pretrained_model_path=config.encoder_pretrained_path,
        max_encoder_sequence_length=config.max_encoder_sequence_length)
    decoder = init_decoder(
        architecture=config.decoder_architecture,
        pretrained_model_path=config.decoder_pretrained_path,
        max_decoder_sequence_length=config.max_decoder_sequence_length,
        num_hidden_layers=config.decoder_num_hidden_layers)
    summarizer = AbsSummarizer(
        encoder=encoder,
        decoder=decoder,
        tokenizer=tokenizer,
        decoder_start_token_id=config.decoder_start_token_id,
        decoder_end_token_id=config.decoder_end_token_id,
        alpha=config.alpha,
        block_trigram=config.block_trigram)
    device = torch.device(f"cuda:{gpu_rank}" if gpu_rank != -1 else "cpu")
    summarizer.to(device)
    if nb_gpu > 0:
        summarizer = DDP(summarizer, device_ids=[device], output_device=device)
    # model initialization />

    # < optimizer and scheduler
    encoder_optimizer, encoder_scheduler = create_optimizers_and_schedulers(
        encoder,
        total_steps=total_steps,
        weight_decay=config.weight_decay,
        learning_rate=config.encoder_learning_rate,
        adam_epsilon=config.adam_epsilon,
        betas=(config.beta1, config.beta2),
        num_warmup_steps=config.num_warmup_steps
    )
    decoder_optimizer, decoder_scheduler = create_optimizers_and_schedulers(
        decoder,
        total_steps=total_steps,
        weight_decay=config.weight_decay,
        learning_rate=config.decoder_learning_rate,
        adam_epsilon=config.adam_epsilon,
        betas=(config.beta1, config.beta2),
        num_warmup_steps=config.num_warmup_steps
    )
    optimizer = AbsSummarizerOptimizer(
        optimizers={'encoder': encoder_optimizer, 'decoder': decoder_optimizer},
        schedulers={'encoder': encoder_scheduler, 'decoder': decoder_scheduler}
    )
    # optimizer and scheduler />

    # < checkpoint restore
    checkpoint_dir = os.path.dirname(config.checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = glob.glob(f"{config.checkpoint_path}*")
    if len(checkpoint_path) > 0:
        checkpoint_path = sorted(checkpoint_path, key=lambda x: os.path.getctime(x), reverse=True)[0]

    if checkpoint_path: # checkpoint found
        logger.info("Loading checkpoint from '{}'.".format(checkpoint_path))
        saved_state = torch.load(checkpoint_path, map_location=lambda s, t: s)
        model_state = saved_state['model']
        summarizer.load_state_dict(model_state)
        optimizer_state = saved_state['optimizer']
        scheduler_state = saved_state['scheduler']
        optimizer.restore(optimizer_state, scheduler_state)
        number_of_updates = saved_state['number_of_updates']
        global_step = saved_state['global_step']
        done_epochs = saved_state['done_epochs']
        done_steps = saved_state['done_steps']
    else:
        number_of_updates = 0
        global_step = 0
        done_epochs = 0
        done_steps = 0

    steps_per_epoch = (len(data_loader.dataset) - 1) // config.batch_size + 1
    assert global_step == (done_epochs * steps_per_epoch + done_steps), \
        "Reproduction checking failed, the training may not be properly resumed."
    # checkpoint restore />

    # < trainer
    trainer = SummarizerTrainer(
        summarizer=summarizer,
        optimizer=optimizer,
        data_loader=data_loader,
        dev_dataloader=dev_data_loader,
        config=config,
        device=device,
        gpu_rank=gpu_rank,
        nb_gpu=nb_gpu,
        done_epochs=done_epochs,
        done_steps=done_steps,
        global_step=global_step,
        number_of_updates=number_of_updates
    )
    trainer.train()
    # trainer />


def run(rank, world_size, config: TrainingConfig, error_queue):
    try:
        dist.init_process_group(backend="nccl", world_size=world_size, rank=rank)
        torch.cuda.set_device(rank)
        # setup logging
        if rank == 0:
            logging.basicConfig(level=logging.INFO)
            add_color_formatter(logging.root)

        setup_and_train(config=config, gpu_rank=rank, nb_gpu=world_size)
    except KeyboardInterrupt:
        pass # killed by parent, do nothing
    except Exception:
        import traceback
        error_queue.put((rank, traceback.format_exc()))


if __name__ == "__main__":
    main()
