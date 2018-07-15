from typing import Union, Tuple, Optional
# noinspection PyPackageRequirements
import torch.nn as nn
# noinspection PyPackageRequirements
from torch.autograd import Variable
# noinspection PyPackageRequirements
from torch.nn import Parameter


# noinspection PyMissingConstructor
class ConvLSTMCell(nn.Module):

    weight_ih: Parameter
    weight_hh: Parameter
    bias_ih: Union[Parameter, None]
    bias_hh: Union[Parameter, None]

    def __init__(self, in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = ...,
                 padding: Union[int, Tuple[int, int]] = ...,
                 dilation: Union[int, Tuple[int, int]] = ...,
                 groups: int = ...,
                 bias: bool = ...,
                 ) -> None: ...

    def reset_parameters(self) -> None: ...

    def forward(self, inputs: Variable,
                hidden: Tuple[Variable, Variable],
                ) -> Tuple[Variable, Tuple[Variable, Variable]]: ...

    @staticmethod
    def _pair(x: Union[int, Tuple[int, int]],
              ) -> Tuple[int, int]: ...


# noinspection PyAbstractClass,PyMissingConstructor
class SatLU(nn.Module):
    def __init__(self, upper: float) -> None: ...
    def forward(self, x: Variable) -> Variable: ...
    def __repr__(self) -> str: ...


# noinspection PyAbstractClass,PyMissingConstructor
class PredNetForwardConv(nn.Module):
    def __init__(self, conv: Union[None, nn.Conv2d, nn.Sequential] = ...,
                 with_error: bool = ...,
                 ) -> None: ...
    def forward(self, x: Optional[Variable]) -> Variable: ...
