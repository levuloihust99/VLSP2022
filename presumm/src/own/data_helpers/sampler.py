import torch
from torch.utils.data.sampler import RandomSampler
from typing import Iterator, Optional, Sized


class ReproducibleRandomSampler(RandomSampler):
    def __init__(self, data_source: Sized, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None, seed=None) -> None:
        super(ReproducibleRandomSampler, self).__init__(data_source, replacement, num_samples, generator)
        self._seed = seed
    
    def seed(self, seed):
        self._seed = seed

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self._seed is None:
            if self.generator is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
                generator = torch.Generator()
                generator.manual_seed(seed)
            else:
                generator = self.generator
        else:
            generator = torch.Generator()
            generator.manual_seed(self._seed)

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        else:
            for _ in range(self.num_samples // n):
                yield from torch.randperm(n, generator=generator).tolist()
            yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]
