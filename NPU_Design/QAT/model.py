import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from config import Q_BITS, Q, Q_noise_type

# ### STE

# In[ ]:


class roundpass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ## Define output w.r.t. input
        output = torch.round(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        ## Define grad_input w.r.t. grad_output
        grad_input = grad_output
        return grad_input


roundpass = roundpass.apply


# ### NIPQ

# In[ ]:


class roundpass_n(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        ## Define output w.r.t. input
        rnd = torch.empty_like(input).uniform_().sub_(0.5)

        output = input + rnd * scale
        ctx.save_for_backward(rnd)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        ## Define grad_input w.r.t. grad_output
        rnd = ctx.saved_tensors

        grad_input = grad_output
        grad_scale = torch.sum(grad_output * rnd)
        return grad_input, grad_scale


roundpass_n = roundpass_n.apply


# ### Quantization module

# In[ ]:


class Quantizer(nn.Module):
    def __init__(self, bits=Q_BITS, always_pos=False):
        super(Quantizer, self).__init__()

        self.first = True

        self.alpha_baseline = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.alpha_delta = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.always_pos = always_pos

        self.Qp = 2 ** (bits - 1) - 1
        self.Qn = -self.Qp
        self.num_steps = self.Qp - self.Qn

    def get_alpha(self):
        return F.softplus(self.alpha_baseline + self.alpha_delta)

    def forward(self, x):
        if self.first:

            def reverse_softplus(x):
                return np.log(np.exp(x) - 1.0)

            self.alpha_baseline.add_(reverse_softplus(x.std().item() * 3))
            self.first = False

        alpha = self.get_alpha()

        step_size = 2 * alpha / self.num_steps

        if self.always_pos:
            off = alpha
        else:
            off = 0

        ## define q_x given x and other components above.
        # STE
        if Q == "STE" or not self.training:
            q_x = (
                torch.clamp(roundpass((x - off) / step_size), self.Qn, self.Qp)
                * step_size
                + off
            )
        # NIPQ
        elif Q == "NIPQ":
            rnd = None
            if Q_noise_type == "uniform":
                rnd = torch.rand_like(x).sub_(0.5)
            elif Q_noise_type == "normal":
                rnd = torch.randn_like(x).mul_(0.5)
            elif Q_noise_type == "normal_rounded":
                rnd = torch.randn_like(x).mul_(0.5).round_()

            q_x = (
                torch.clamp(
                    (x - off) + rnd * step_size,
                    self.Qn * step_size,
                    self.Qp * step_size,
                )
                + off
            )
        return q_x


# ### Quantization aware modules

# In[ ]:


class CustomConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(CustomConv2d, self).__init__(*args, **kwargs)
        self.q_w = Quantizer()
        self.q_a = Quantizer(always_pos=True)
        self.is_quant = False  # No quantization by default

    def forward(self, x):
        if self.is_quant:
            ## quantize the weights and inputs using the ``Quantize`` modules.
            weight = self.q_w(self.weight)
            inputs = self.q_a(x)
        else:
            weight = self.weight
            inputs = x

        return F.conv2d(
            inputs,
            weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class CustomLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(CustomLinear, self).__init__(*args, **kwargs)
        self.q_w = Quantizer()
        self.q_a = Quantizer(always_pos=True)
        self.is_quant = False  # No quantization by default

    def forward(self, x):
        if self.is_quant:
            ## quantize the weights and inputs using the ``Quantize`` modules.
            weight = self.q_w(self.weight)
            inputs = self.q_a(x)
        else:
            weight = self.weight
            inputs = x

        return F.linear(inputs, weight, bias=self.bias)


# ### neural network

# In[ ]:


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv = CustomConv2d(
            1, 6, kernel_size=3, stride=1, padding=0, bias=False
        )
        self.layer1 = nn.Sequential(
            self.conv,
            nn.BatchNorm2d(6),
            nn.ReLU(),
        )

        self.fc1 = CustomLinear(4056, 128, bias=False)
        self.fc2 = CustomLinear(128, 10, bias=False)
        self.layer2 = nn.Sequential(self.fc1, torch.nn.ReLU(), self.fc2)

    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.size(0), -1)
        out = self.layer2(out)
        return out
