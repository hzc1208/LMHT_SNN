import torch
import torch.nn as nn
from typing import Tuple, Union

def unpack_for_conv(x: Union[Tuple[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    if isinstance(x, tuple):
        assert x.__len__() == 1
        x = x[0]
    if len(x.shape) == 5:
        return x.flatten(0, 1)
    return x


def unpack_for_linear(x: Union[Tuple[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    if isinstance(x, tuple):
        assert x.__len__() == 1
        x = x[0]
    if len(x.shape) == 3:
        return x.flatten(0, 1)
    return x
    

class BaseMonitor:
    def __init__(self):
        self.hooks = []
        self.monitored_layers = []
        self.records = []
        self.name_records_index = {}
        self._enable = True

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.records[i]
        elif isinstance(i, str):
            y = []
            for index in self.name_records_index[i]:
                y.append(self.records[index])
            return y
        else:
            raise ValueError(i)

    def clear_recorded_data(self):
        self.records.clear()
        for k, v in self.name_records_index.items():
            v.clear()

    def enable(self):
        self._enable = True

    def disable(self):
        self._enable = False

    def is_enable(self):
        return self._enable

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def __del__(self):
        self.remove_hooks()


class SOPMonitor(BaseMonitor):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.skip_name = ['conv1']
        for name, m in net.named_modules():
            if name in self.skip_name:
                continue
            if isinstance(m, nn.Conv2d):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(
                    self.create_hook_conv(name)))
            elif isinstance(m, nn.Linear):
                self.monitored_layers.append(name)
                self.name_records_index[name] = []
                self.hooks.append(m.register_forward_hook(
                    self.create_hook_linear(name)))


    def cal_sop_conv(self, x: torch.Tensor, m: nn.Conv2d):
        with torch.no_grad():
            out = torch.nn.functional.conv2d(x, torch.ones_like(m.weight), None, m.stride, m.padding, m.dilation, m.groups)
            return out.sum().unsqueeze(0)

    def create_hook_conv(self, name):
        def hook(m: nn.Conv2d, x: torch.Tensor, y: torch.Tensor):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(self.cal_sop_conv(unpack_for_conv(x).detach(), m))

        return hook

    def cal_sop_linear(self, x: torch.Tensor, m: nn.Linear):
        with torch.no_grad():
            out = torch.nn.functional.linear(x, torch.ones_like(m.weight), None)
            return out.sum().unsqueeze(0)

    def create_hook_linear(self, name):
        def hook(m: nn.Conv2d, x: torch.Tensor, y: torch.Tensor):
            if self.is_enable():
                self.name_records_index[name].append(self.records.__len__())
                self.records.append(self.cal_sop_linear(unpack_for_linear(x).detach(), m))

        return hook


def l_prod(in_list):
    res = 1
    for _ in in_list:
        res *= _
    return res


def calculate_conv2d_flops(input_size: list, output_size: list, kernel_size: list, groups: int):
    in_c = input_size[1]
    g = groups
    return l_prod(output_size) * (in_c // g) * l_prod(kernel_size[2:])


def count_convNd(m, x, y: torch.Tensor):
    if isinstance(x, tuple):
        assert x.__len__() == 1
        x = x[0]    
    m.total_ops += calculate_conv2d_flops(input_size=list(x.shape), output_size=list(y.shape), kernel_size=list(m.weight.shape), groups=m.groups)


def count_linear(m, x, y):
    total_mul = m.in_features
    num_elements = y.numel()
    m.total_ops += torch.DoubleTensor([int(total_mul * num_elements)])
    
