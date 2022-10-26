import copy
import argparse
from omegaconf import DictConfig
from typing import Union, Any


def recursive_apply(data, fn):
    stack = [(None, -1, data)]  # parent, idx, child: parent[idx] = child
    while stack:
        parent_node, index, node = stack.pop()
        if isinstance(node, list):
            stack.extend(list(zip([node] * len(node), range(len(node)), node)))
        elif isinstance(node, dict):
            stack.extend(
                list(zip([node] * len(node), node.keys(), node.values())))
        elif isinstance(node, str):
            if node:
                parent_node[index] = fn(node)
            else:
                parent_node[index] = node
        else:
            continue
    return data


def dictconfig_to_namespace(cfg: Union[DictConfig, Any]):
    if not isinstance(cfg, DictConfig):
        return copy.deepcopy(cfg)

    args = argparse.Namespace()
    for attr_name in dir(cfg):
        attr = getattr(cfg, attr_name)
        setattr(args, attr_name, dictconfig_to_namespace(attr))

    return args
