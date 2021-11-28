# ====================================================
#   Copyright (C) 2021  All rights reserved.
#
#   Author        : Xinyu Zhu
#   Email         : zhuxy21@mails.tsinghua.edu.cn
#   File Name     : optimization.py
#   Last Modified : 2021-11-19 15:45
#   Describe      : modified from https://github.com/MuQiuJun-AI/bert4pytorch/blob/master/bert4pytorch/optimization.py
#
# ====================================================

import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.distributions.bernoulli import Bernoulli
from typing import Callable, Iterable, Optional, Tuple, Union


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    线性warmup的schedule，前num_warmup_steps个step学习率从0线性增长到lr，然后一直衰减至训练结束时达到0

    参数
        num_warmup_steps：
            需要warmup的步数，等于 num_training_steps * warmup_proportion(warmup的比例，建议0.05-0.15)

        num_training_steps:
            总的训练步数，一般为 num_batch * num_epoch
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class AdamW(Optimizer):
    """
    带权重衰减的Adam，同时集成了Child Tuning功能
    <https://arxiv.org/abs/1711.05101>`__.
    <https://github.com/alibaba/AliceMind/blob/c17b177e7e/ChildTuning/ChildTuningOptimizer.py>

    参数:
        params (:obj:`Iterable[torch.nn.parameter.Parameter]`):
        lr (:obj:`float`, `optional`, defaults to 1e-3):
            学习率.
        betas (:obj:`Tuple[float,float]`, `optional`, defaults to (0.9, 0.999)):
            Adam的betas参数 (b1, b2)
        eps (:obj:`float`, `optional`, defaults to 1e-6):
            Adam的epsilon参数，用于数值稳定性
        weight_decay (:obj:`float`, `optional`, defaults to 0):
            权重衰减参数
        correct_bias (:obj:`bool`, `optional`, defaults to `True`):
            修正Adam的bias (原始的tf版本的bert，没有修正bias，取值为False，但是可以尝试用True，可能会收敛更稳定)
        reserve_p (:obj:`float`, `optional`, defaults to 1.0):
            使用child tuning时对grad进行dropout的p值，0.9代表随机 drop 10% 的grad，默认不使用Child Tuning
    例子:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5, correct_bias=False, reserve_p=0.9)

    """

    def __init__(
        self,
        params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        reserve_p: float = 1.0
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0]")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0]")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        self.reserve_p = reserve_p
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        """
        执行单步优化

        参数:
            closure (:obj:`Callable`, `optional`): 
                评估模型并返回loss，是一个闭包
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")
                
                # ChildTuning-Free
                # =================== HACK BEGIN =======================         
                grad_mask = Bernoulli(grad.new_full(size=grad.size(), fill_value=self.reserve_p))
                grad *= grad_mask.sample() / self.reserve_p
                # =================== HACK END =======================
                state = self.state[p]

                # state初始化
                if len(state) == 0:
                    state["step"] = 0
                    # 一阶梯度的指数加权移动平均，也即累积一阶动量的计算
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # 二阶梯度的指数加权移动平均，也即累积二阶动量的计算
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # 计算一二阶梯度的beta系数下的衰减值，并进行更新
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                # 修正bias，因为开始时一阶梯度和二阶梯度都太小。
                if group["correct_bias"]:  
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # 直接给 loss 中加上参数权重的l2范数平方并不是一个在Adam中正确使用L2正则的方式，
                # 因为这会让一阶和二阶梯度m和v变得很奇怪，这种相互作用会导致Adam的L2正则表现不佳，
                # 具体来说，两个L2范数一样的weight，因为二阶梯度对梯度的归一化，梯度更大的那个衰减得更慢，
                # 而使用权重衰减，不与m和v直接交互，让每个梯度都以相同的比例进行衰减，这等价于SGD下的L2正则
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss
    

class ExponentialMovingAverage():
    """
        模型权重的指数滑动平均
        注意区别于类似adam一类的自适应学习率优化器，adam是针对一阶二阶梯度的指数滑动平均，
        而该EMA本质并不是一个优化器，而是在训练过程中对整个模型参数的指数滑动平均，两者完全不同

        例子:
            # 初始化
            ema = ExponentialMovingAverage(model, 0.999)

            # 训练过程中，更新完参数后，同步update ema_weights weights
            def train():
                optimizer.step()
                ema.update()

            # eval前，调用apply_ema_weights weights；eval之后，恢复原来模型的参数
            def evaluate():
                ema.apply_ema_weights()
                # evaluate
                # 如果想保存ema后的模型，请在reset_old_weights方法之前调用torch.save()
                ema.reset_old_weights()
    """
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        # 保存ema权重（当前step的每一层的滑动平均权重）
        self.ema_weights = {}
        # 在进行evaluate的时候，保存原始的模型权重，当执行完evaluate后，从ema权重恢复到原始权重
        self.model_weights = {}

        # 初始化ema_weights为model_weights
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.ema_weights[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.ema_weights
                new_average = (1.0 - self.decay) * param.data + self.decay * self.ema_weights[name]
                self.ema_weights[name] = new_average.clone()
    
    def apply_ema_weights(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.ema_weights
                self.model_weights[name] = param.data
                param.data = self.ema_weights[name]
    
    def reset_old_weights(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.model_weights
                param.data = self.model_weights[name]
        self.model_weights = {}
