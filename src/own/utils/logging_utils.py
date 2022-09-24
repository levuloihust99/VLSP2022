import os
import logging

BOLD_GREEN = "\x1b[38;5;2;1m"
RESET_CODE = "\x1b[0m"


class ColorFormatter(logging.Formatter):

    FORMAT_STR = BOLD_GREEN + "%(asctime)s - %(name)s - %(levelname)s: " + RESET_CODE + "%(message)s"

    def format(self, record):
        formatter = logging.Formatter(self.FORMAT_STR)
        return formatter.format(record)


def add_color_formater(logger: logging.Logger):
    handlers = logger.handlers
    for handler in handlers:
        handler.setFormatter(ColorFormatter())
