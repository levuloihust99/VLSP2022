import copy
from typing import Union
from omegaconf import DictConfig, ListConfig


def convert_dictconfig_to_dict(cfg: Union[DictConfig, ListConfig]):
    if isinstance(cfg, DictConfig):
        ret = {}
        for k, v in cfg.items():
            if isinstance(v, (DictConfig, ListConfig)):
                ret[k] = convert_dictconfig_to_dict(cfg[k])
            else:
                ret[k] = v
    elif isinstance(cfg, ListConfig):
        ret = []
        for idx, v in enumerate(cfg):
            if isinstance(v, (DictConfig, ListConfig)):
                cfg[idx] = convert_dictconfig_to_dict(v)
            else:
                cfg[idx] = v
    else:
        raise Exception("Input of 'convert_dictconfig_to_dict' must be of type DictConfig or ListConfig. "
            "Encounter type {}".format(type(cfg)))
    return ret
