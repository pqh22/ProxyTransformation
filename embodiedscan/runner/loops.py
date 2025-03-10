# Copyright (c) OpenMMLab. All rights reserved.
import bisect
import logging
import time
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union
from itertools import islice

import torch.distributed as dist
from torch.utils.data import DataLoader

import mmengine
from mmengine.logging import print_log
from mmengine.runner import IterBasedTrainLoop
from embodiedscan.registry import LOOPS
from tqdm import tqdm


@LOOPS.register_module()
class FastResumeIterBasedTrainLoop(IterBasedTrainLoop):
    """Loop for iter-based training.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        max_iters (int): Total training iterations.
        val_begin (int): The iteration that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1000.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """

    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_iters: int,
            val_begin: int = 1,
            val_interval: int = 1000,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(runner, dataloader, max_iters, val_begin, val_interval, dynamic_intervals)

    def run(self) -> None:
        """Launch training."""
        self.runner.call_hook('before_train')
        # In iteration-based training loop, we treat the whole training process
        # as a big epoch and execute the corresponding hook.
        self.runner.call_hook('before_train_epoch')
        if os.getenv('PDB_DEBUG', '0') == '1':
            import pdb; pdb.set_trace()
        if self._iter > 0:
            print_log(
                f'Advance dataloader {self._iter} steps to skip data '
                'that has already been trained',
                logger='current',
                level=logging.WARNING)
            islice(self.dataloader_iterator, 0, self._iter)
                # for _ in tqdm(range(self._iter)):
                #     next(self.dataloader_iterator)
            print_log(
                f'Done advancing dataloader.',
                logger='current',
                level=logging.WARNING)

        while self._iter < self._max_iters and not self.stop_training:
            self.runner.model.train()

            data_batch = next(self.dataloader_iterator)
            self.run_iter(data_batch)

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._iter >= self.val_begin
                    and (self._iter % self.val_interval == 0
                         or self._iter == self._max_iters)):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train_epoch')
        self.runner.call_hook('after_train')
        return self.runner.model