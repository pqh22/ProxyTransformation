import logging
import math
from abc import abstractmethod
from typing import Optional, Union

import torch
import torch.nn as nn
from mmengine.logging import print_log
from torch import Tensor
from copy import deepcopy

from embodiedscan.registry import MODELS

class BaseAveragedModel(nn.Module):
    """A base class for averaging model weights.

    Weight averaging, such as SWA and EMA, is a widely used technique for
    training neural networks. This class implements the averaging process
    for a model. All subclasses must implement the `avg_func` method.
    This class creates a copy of the provided module :attr:`model`
    on the :attr:`device` and allows computing running averages of the
    parameters of the :attr:`model`.

    The code is referenced from: https://github.com/pytorch/pytorch/blob/master/torch/optim/swa_utils.py.

    Different from the `AveragedModel` in PyTorch, we use in-place operation
    to improve the parameter updating speed, which is about 5 times faster
    than the non-in-place version.

    In mmengine, we provide two ways to use the model averaging:

    1. Use the model averaging module in hook:
       We provide an :class:`mmengine.hooks.EMAHook` to apply the model
       averaging during training. Add ``custom_hooks=[dict(type='EMAHook')]``
       to the config or the runner.

    2. Use the model averaging module directly in the algorithm. Take the ema
       teacher in semi-supervise as an example:

       >>> from mmengine.model import ExponentialMovingAverage
       >>> student = ResNet(depth=50)
       >>> # use ema model as teacher
       >>> ema_teacher = ExponentialMovingAverage(student)

    Args:
        model (nn.Module): The model to be averaged.
        interval (int): Interval between two updates. Defaults to 1.
        device (torch.device, optional): If provided, the averaged model will
            be stored on the :attr:`device`. Defaults to None.
        update_buffers (bool): if True, it will compute running averages for
            both the parameters and the buffers of the model. Defaults to
            False.
    """  # noqa: E501

    def __init__(self,
                 model: nn.Module,
                 interval: int = 1,
                 device: Optional[torch.device] = None,
                 update_buffers: bool = False,
                 model_cfg_dict: dict = None) -> None:
        super().__init__()
        if model_cfg_dict is not None:
            self.module = MODELS.build(model_cfg_dict).requires_grad_(False)
        else:
            self.module = deepcopy(model).requires_grad_(False)
        self.interval = interval
        if device is not None:
            self.module = self.module.to(device)
        self.register_buffer('steps',
                             torch.tensor(0, dtype=torch.long, device=device))
        self.update_buffers = update_buffers
        if update_buffers:
            self.avg_parameters = self.module.state_dict()
        else:
            self.avg_parameters = dict(self.module.named_parameters())

    @abstractmethod
    def avg_func(self, averaged_param: Tensor, source_param: Tensor,
                 steps: int) -> None:
        """Use in-place operation to compute the average of the parameters. All
        subclasses must implement this method.

        Args:
            averaged_param (Tensor): The averaged parameters.
            source_param (Tensor): The source parameters.
            steps (int): The number of times the parameters have been
                updated.
        """

    def forward(self, *args, **kwargs):
        """Forward method of the averaged model."""
        return self.module(*args, **kwargs)

    def update_parameters(self, model: nn.Module) -> None:
        """Update the parameters of the model. This method will execute the
        ``avg_func`` to compute the new parameters and update the model's
        parameters.

        Args:
            model (nn.Module): The model whose parameters will be averaged.
        """
        src_parameters = (
            model.state_dict()
            if self.update_buffers else dict(model.named_parameters()))
        if self.steps == 0:
            for k, p_avg in self.avg_parameters.items():
                p_avg.data.copy_(src_parameters[k].data)
        elif self.steps % self.interval == 0:
            for k, p_avg in self.avg_parameters.items():
                if p_avg.dtype.is_floating_point:
                    device = p_avg.device
                    self.avg_func(p_avg.data,
                                  src_parameters[k].data.to(device),
                                  self.steps)
        if not self.update_buffers:
            # If not update the buffers,
            # keep the buffers in sync with the source model.
            for b_avg, b_src in zip(self.module.buffers(), model.buffers()):
                b_avg.data.copy_(b_src.data.to(b_avg.device))
        self.steps += 1


@MODELS.register_module()
class ExpMomentumEMA(BaseAveragedModel):
    """Exponential moving average (EMA) with exponential momentum strategy,
    which is used in YOLOX.

    Args:
        model (nn.Module): The model to be averaged.
        momentum (float): The momentum used for updating ema parameter.
            Ema's parameter are updated with the formula:
           `averaged_param = (1-momentum) * averaged_param + momentum *
           source_param`. Defaults to 0.0002.
        gamma (int): Use a larger momentum early in training and gradually
            annealing to a smaller value to update the ema model smoothly. The
            momentum is calculated as
            `(1 - momentum) * exp(-(1 + steps) / gamma) + momentum`.
            Defaults to 2000.
        interval (int): Interval between two updates. Defaults to 1.
        device (torch.device, optional): If provided, the averaged model will
            be stored on the :attr:`device`. Defaults to None.
        update_buffers (bool): if True, it will compute running averages for
            both the parameters and the buffers of the model. Defaults to
            False.
    """

    def __init__(self,
                 model: nn.Module,
                 momentum: float = 0.0002,
                 gamma: int = 2000,
                 interval=1,
                 device: Optional[torch.device] = None,
                 update_buffers: bool = False,
                 model_cfg_dict: dict = None) -> None:
        super().__init__(
            model=model,
            interval=interval,
            device=device,
            update_buffers=update_buffers,
            model_cfg_dict=model_cfg_dict)
        
        assert 0.0 < momentum < 1.0, 'momentum must be in range (0.0, 1.0)'\
                                     f'but got {momentum}'
        if momentum > 0.5:
            print_log(
                'The value of momentum in EMA is usually a small number,'
                'which is different from the conventional notion of '
                f'momentum but got {momentum}. Please make sure the '
                f'value is correct.',
                logger='current',
                level=logging.WARNING)
        self.momentum = momentum
            
        assert gamma > 0, f'gamma must be greater than 0, but got {gamma}'
        self.gamma = gamma

    def avg_func(self, averaged_param: Tensor, source_param: Tensor,
                 steps: int) -> None:
        """Compute the moving average of the parameters using the exponential
        momentum strategy.

        Args:
            averaged_param (Tensor): The averaged parameters.
            source_param (Tensor): The source parameters.
            steps (int): The number of times the parameters have been
                updated.
        """
        momentum = (1 - self.momentum) * math.exp(
            -float(1 + steps) / self.gamma) + self.momentum
        averaged_param.lerp_(source_param, momentum)