import torch.nn as nn
import numpy


class BinOp():

    def __init__(self, model):

        count_conv2d = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                count_conv2d += 1

        start_range = 1
        end_range = count_conv2d - 2
        self.bin_range = numpy.linspace(start_range, end_range, end_range-start_range+1)\
                    .astype('int').tolist()

        self.num_params = len(self.bin_range)
        self.saved_params = []
        self.target_modules = []

        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                index += 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)

    def binarization(self):
        self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.binarizeConvParams()

    def meancenterConvParams(self):
        for index in range(self.num_params):
            neg_mean = self.target_modules[index].data.mean(1, keepdim=True).mul(-1)\
                .expand_as(self.target_modules[index].data)
            self.target_modules[index].data.add_(neg_mean)

    def clampConvParams(self):
        for index in range(self.num_params):
            self.target_modules[index].data.clamp_(-1.0, 1.0)

    def save_params(self):
        for index in range(self.num_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarizeConvParams(self):
        for index in range(self.num_params):
            n = self.target_modules[index].data[0].nelement()
            s = self.target_modules[index].data.size()
            alpha = self.target_modules[index].data.norm(1, dim=[1, 2, 3], keepdim=True)\
                .div(n).expand(s)
            self.target_modules[index].data = self.target_modules[index].data.sign()\
                .mul(alpha)

    def restore(self):
        for index in range(self.num_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def updateBinaryWeightGrad(self):
        for index in range(self.num_params):
            weight = self.saved_params[index]
            n = weight[0].nelement()
            s = weight.size()
            m = weight.norm(1, dim=[1, 2, 3], keepdim=True).div(n).expand(s)
            m[weight.lt(-1.0)] = 0
            m[weight.gt(1.0)] = 0
            m = m.mul(weight).mul(self.target_modules[index].grad.data)
            m_add = weight.sign().mul(self.target_modules[index].grad.data)
            m_add = m_add.sum(dim=[1, 2, 3], keepdim=True).div(n).expand(s)
            m_add = m_add.mul(weight.sign())
            self.target_modules[index].grad.data = m.add(m_add)
