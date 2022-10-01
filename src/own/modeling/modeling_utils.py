import torch


def tile(x, count, dim=0):
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .repeat(1, count) \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def recursive_apply(x, fn):
    if isinstance(x, dict):
        return {k: recursive_apply(v, fn) for k, v in x.items()}
    elif isinstance(x, (tuple, list)):
        return x.__class__([recursive_apply(e, fn) for e in x])
    else:
        assert isinstance(x, torch.Tensor)
        return fn(x)
