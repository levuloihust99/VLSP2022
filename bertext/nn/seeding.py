def setup_seeding(seed: int):
    import random
    import numpy as np
    import torch
    from torch._C import default_generator

    random.seed(seed)
    np.random.seed(seed)

    default_generator.manual_seed(seed)
    torch.cuda.manual_seed(seed)
