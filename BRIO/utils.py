import os
import torch
import logging

from tensorboardX import SummaryWriter
from datetime import datetime

logger = logging.getLogger(__name__)


class Recorder():
    def __init__(self, id, log=True):
        self.log = log
        now = datetime.now()
        date = now.strftime("%y-%m-%d")
        self.dir = f"./cache/{date}-{id}"
        if self.log:
            os.mkdir(self.dir)
            self.f = open(os.path.join(self.dir, "log.txt"), "w")
            self.writer = SummaryWriter(os.path.join(self.dir, "log"), flush_secs=60)
        
    def write_config(self, args, models, name):
        if self.log:
            with open(os.path.join(self.dir, "config.txt"), "w") as f:
                print(name, file=f)
                print(args, file=f)
                print(file=f)
                for (i, x) in enumerate(models):
                    print(x, file=f)
                    print(file=f)
        logger.info("{}\n".format(args))
        for (i, x) in enumerate(models):
            logger.info("{}\n".format(x))

    def print(self, x=None):
        if x is not None:
            logger.info(x)
        if self.log:
            if x is not None:
                print(x, file=self.f, flush=True)
            else:
                print(file=self.f, flush=True)

    def plot(self, tag, values, step):
        if self.log:
            self.writer.add_scalars(tag, values, step)
    
    def write_log(self, logs, step):
        for k, v in logs.items():
            self.writer.add_scalar(k, v, step)

    def __del__(self):
        if self.log:
            self.f.close()
            self.writer.close()

    def save(self, model, name):
        if self.log:
            torch.save(model.state_dict(), os.path.join(self.dir, name))
