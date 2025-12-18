#!/usr/bin/env python3

from distutils.errors import DistutilsModuleError
import os
import argparse
import numpy as np
import cv2
from pathlib import Path
import copy
import logging
import random
import scipy.io as sio
import time
import math

import torch
from fvcore.nn import FlopCountAnalysis
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from srm_filter_kernel import all_normalized_hpf_list

from collections import defaultdict, deque
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


TRAIN_FILE_COUNT = 14000
TRAIN_PRINT_FREQUENCY = 100
EVAL_PRINT_FREQUENCY = 1

OUTPUT_PATH = Path(__file__).stem


class TLU(nn.Module):
    def __init__(self, threshold):
        super(TLU, self).__init__()

        self.threshold = threshold

    def forward(self, input):
        output = torch.clamp(input, min=-self.threshold, max=self.threshold)

        return output


def build_optimizer(model, args):
    decay, no_decay = [], []
    for m in model.modules():
        for n, p in m.named_parameters(recurse=False):
            if not p.requires_grad: 
                continue
            # WD 제외 조건: bias 이거나, 정규화 레이어의 파라미터면 no_decay
            if n.endswith('bias') or isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
                no_decay.append(p)
            else:
                decay.append(p)

    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            [{'params': decay,    'weight_decay': args.weight_decay},
             {'params': no_decay, 'weight_decay': 0.0}],
            lr=args.lr)
    else:  # sgd
        optimizer = torch.optim.SGD(
            [{'params': decay,    'weight_decay': args.weight_decay},
             {'params': no_decay, 'weight_decay': 0.0}],
            lr=args.lr, momentum=args.momentum, nesterov=True)
    return optimizer


def build_filters():
    filters = []
    ksize = [5]
    lamda = np.pi / 2.0
    sigma = [0.5, 1.0]
    phi = [0, np.pi / 2]
    for hpf_item in all_normalized_hpf_list: # 원래 크기가 3×3, 4×4 등인 SRM 필터를 5×5로 zero‑padding / SRM 고역통과필터 개수(30개)
       row_1 = int((5 - hpf_item.shape[0]) / 2)
       row_2 = int((5 - hpf_item.shape[0]) - row_1)
       col_1 = int((5 - hpf_item.shape[1]) / 2)
       col_2 = int((5 - hpf_item.shape[1]) - col_1)
       hpf_item = np.pad(hpf_item, pad_width=((row_1, row_2), (col_1, col_2)), mode='constant')
       filters.append(hpf_item)
    for theta in np.arange(0, np.pi, np.pi / 8):  # gabor 0 22.5 45 67.5 90 112.5 135 157.5 / Gabor 필터 (8방향 × 2대역폭 × 2위상)
        # theta가 0부터 π까지 π/8 간격으로 8개 / sigma가 두 값 (0.5, 1.0) / phi가 두 값 (0, π/2) → 8 × 2 × 2 = 32개의 Gabor 커널이 생성되어 추가
        for k in range(2):
            for j in range(2):
                kern = cv2.getGaborKernel((ksize[0], ksize[0]), sigma[k], theta, sigma[k] / 0.56, 0.5, phi[j],
                                          ktype=cv2.CV_32F)
                filters.append(kern)
    return filters # len(filters) == len(all_normalized_hpf_list) + 32 == 62


def filter_meta():
    """
    index -> ('SRM', {}) or ('Gabor', {'theta','sigma','phi'})
    build_filters()와 정확히 동일한 순서
    """
    meta = []
    # SRM 30개
    for _ in range(len(all_normalized_hpf_list)):
        meta.append(("SRM", {}))
    # Gabor 32개 (θ: 0~π, step π/8 / σ: {0.5,1.0} / φ: {0,π/2})
    for theta in np.arange(0, np.pi, np.pi/8):
        for sigma in (0.5, 1.0):
            for phi in (0.0, math.pi/2):
                meta.append(("Gabor", {"theta": float(theta), "sigma": float(sigma), "phi": float(phi)}))
    return meta


def pick_gabor_balanced(energy, meta, gabor_quota,
                        sigma_list=(0.5, 1.0), phi_list=(0.0, math.pi/2)):
    """
    우선순위: θ 엄격 균등 → φ 균형 → σ는 에너지 우선(소프트 균형)
    """
    # θ별 후보 상자: 각 θ에 대해 (score, idx, σ, φ) 내림차순
    buckets = defaultdict(list)
    for i, (kind, info) in enumerate(meta):
        if kind != "Gabor": 
            continue
        th, sg, ph = info["theta"], info["sigma"], info["phi"]
        buckets[th].append((float(energy[i].item()), i, sg, ph))
    thetas = sorted(buckets.keys())
    for th in thetas:
        buckets[th].sort(key=lambda x: -x[0])  # 에너지 내림차순
    deques = {th: deque(buckets[th]) for th in thetas}

    # 글로벌 균형 지표(φ는 강하게 균형, σ는 소프트)
    phi_count = {ph: 0 for ph in phi_list}
    selected = []

    while len(selected) < gabor_quota and any(len(dq) > 0 for dq in deques.values()):
        for th in thetas:
            if len(selected) >= gabor_quota:
                break
            dq = deques[th]
            if not dq:
                continue

            # 덜 선택된 φ 우선
            phi_order = sorted(phi_list, key=lambda ph: phi_count[ph])

            # 큐에서 에너지 순으로 보되, 가능한 한 덜 뽑힌 φ를 먼저 선택
            picked = None
            temp = []
            while dq and picked is None:
                cand = dq.popleft()  # (score, idx, sg, ph)
                temp.append(cand)
                _, _, sg, ph = cand
                if ph == phi_order[0]:
                    picked = cand

            if picked is None: # 균형이 불가능하면 에너지 1순위
                picked = temp[0]
                temp = temp[1:]
            # 남은 후보 복원
            for x in reversed(temp):
                dq.appendleft(x)

            # 최종 채택
            if picked is not None:
                try:
                    dq.remove(picked)
                except ValueError:
                    pass
                _, idx, sg, ph = picked
                selected.append(idx)
                phi_count[ph] += 1

            if len(selected) >= gabor_quota:
                break

    return selected


def pick_hpf_mixed(energy, meta, total_k=31, gabor_k=None):
    """
    total_k개 중 Gabor=gabor_k, 나머지는 SRM에서 에너지 상위 선택
    gabor_k가 None이면 전체 비율(32/62)에 맞춤
    """
    srm_idx_all   = [i for i, (k, _) in enumerate(meta) if k == "SRM"]
    gabor_idx_all = [i for i, (k, _) in enumerate(meta) if k == "Gabor"]
    srm_n, gabor_n = len(srm_idx_all), len(gabor_idx_all)  # 30, 32

    if gabor_k is None:
        gabor_k = int(math.ceil(total_k * (gabor_n / (srm_n + gabor_n))))  # 예: 31→16, 15→8
    gabor_k = max(0, min(gabor_k, total_k))
    srm_k   = total_k - gabor_k

    # Gabor는 균형 선택
    gabor_idx = pick_gabor_balanced(energy, meta, gabor_k)

    # SRM은 에너지 상위
    srm_sorted = sorted(srm_idx_all, key=lambda i: -float(energy[i].item()))
    srm_idx = srm_sorted[:srm_k]

    return srm_idx + gabor_idx


def learnable_select_indices(loader, device, bank_size, top_k = 31, epochs = 3, lr = 1e-3, l1_lambda = 1e-4, gabor_k = None) -> list:
    model = HPFSelector(bank_size=bank_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    gate_sum = torch.zeros(bank_size, device=device)
    cnt = 0

    model.train()
    for ep in range(1, epochs+1):
        for sample in loader:
            data, label = sample['data'].to(device), sample['label'].to(device)   # (B,2,3,H,W)
            x = data.view(-1, data.size(2), data.size(3), data.size(4))           # (2B,3,H,W)
            y = x[:, 0:1]                                                         # Y 채널
            lbl = label.view(-1)

            logits, gate, z = model(y)
            ce = F.cross_entropy(logits, lbl)
            l1 = gate.mean()
            loss = ce + l1_lambda * l1

            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

            gate_sum += gate.detach().sum(dim=0)
            cnt += y.size(0)

    # ★ 학습된 점수로 중요도 산출
    score = gate_sum / max(1, cnt)
    score = score.detach().cpu()

    # 단순 top-K (균형 미사용)
    # score = gate_sum / max(1, cnt)
    # idxs  = torch.topk(score, k=top_k, largest=True).indices.tolist()
    # fixed_indices = idxs

    # 균형 선택 사용(SRM/Gabor 비율 + θ/φ 균형)
    meta = filter_meta()[:bank_size]
    fixed_indices = pick_hpf_mixed(score, meta, total_k=top_k, gabor_k=gabor_k)

    return fixed_indices


class HPF(nn.Module):
    def __init__(self, bank_size, fixed_indices=None):
        super().__init__()
        full_list = build_filters()

        if fixed_indices is None:
            sel = list(range(bank_size))
        else:
            sel = list(map(int, fixed_indices))
            bank_size = len(sel)  # 실제 뱅크 크기 = 선택 개수

        self.bank_size = bank_size
        self.fixed_indices = sel

        filt_list = [full_list[i] for i in sel]
        w = torch.tensor(np.stack(filt_list, 0), dtype=torch.float32).view(bank_size, 1, 5, 5)

        self.hpf = nn.Conv2d(1, bank_size, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = nn.Parameter(w, requires_grad=False)
        self.tlu = TLU(5.0)

    def forward(self, x):
        # (B,1,H,W) -> (B,n_filters,H,W)
        x = self.hpf(x)
        x = self.tlu(x)
        return x


class HPFSelector(nn.Module):
    """
    딥러닝 게이트로 '어떤 HPF 채널이 discriminative 한가'를 학습해 Top-K를 뽑아준다.
    구성: HPF(고정) -> GAP -> 작은 MLP(은행별 점수) -> sparsity L1
    """
    def __init__(self, bank_size: int, hidden: int = 128):
        super().__init__()
        self.bank_size = bank_size
        self.hpf = HPF(bank_size=bank_size, fixed_indices=list(range(bank_size)))  # 62개 전부
        # GAP로 (B,bank,H,W) -> (B,bank)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        # 채널 중요도 스코어(게이트) 산출용 MLP
        self.mlp = nn.Sequential(
            nn.Linear(bank_size, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, bank_size, bias=True)  # raw score
        )
        # 보조 분류기(게이트 * z 로 2-class 분류해 지도신호로 학습)
        self.cls = nn.Linear(bank_size, 2, bias=True)

    def forward(self, y):  # y: (B,1,H,W)  (Y채널)
        with torch.no_grad():
            feat = self.hpf(y)                     # (B,bank,H,W)  고정 HPF
        z = self.gap(feat).flatten(1)              # (B,bank), 평균 에너지
        s = self.mlp(z)                            # (B,bank)  중요도 raw score
        gate = torch.sigmoid(s)                    # (B,bank)  0~1
        pooled = gate * z                          # (B,bank)  gated feature
        logits = self.cls(pooled)                  # (B,2)
        return logits, gate, z                     # 학습/통계용 반환


class ConvBNAct(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1,
                 groups=1, norm_layer=nn.BatchNorm2d, act='prelu', leaky_slope=0.10):
        padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding,
                      groups=groups, bias=False),
            norm_layer(out_planes),
        ]
        if act == 'prelu':
            layers.append(nn.PReLU(num_parameters=out_planes, init=0.25))
        elif act == 'leakyrelu':
            layers.append(nn.LeakyReLU(negative_slope=leaky_slope, inplace=True))
        elif act == 'relu6':
            layers.append(nn.ReLU6(inplace=True))
        elif act == 'silu':
            layers.append(nn.SiLU(inplace=True))   # = Swish
        else:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class InvertedResidual(nn.Module):
    """
    MobileNetV2 original block
    [1x1 expand] → [3x3 depthwise + ReLU6 or PReLU] → [1x1 linear project]
    """
    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int, norm_layer=nn.BatchNorm2d, act='prelu', leaky_slope=0.10):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        # stride는 반드시 1 또는 2이어야 하므로 조건 걸기
        assert stride in [1, 2]
        # expansion factor를 이용하여 channel을 확장
        hidden_dim = int(round(inp * expand_ratio))
        # stride가 1인 경우에만 residual block을 사용
        # skip connection을 사용하는 경우 input과 output의 크기가 같아야함
        self.use_res_connect = (self.stride == 1) and (inp == oup)

        # Inverted Residual 연산
        layers = []
        if expand_ratio != 1:
            # point-wise convolution
            layers.append(ConvBNAct(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, act=act, leaky_slope=leaky_slope))
        layers.extend([
            # depth-wise convolution
            ConvBNAct(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer, act=act, leaky_slope=leaky_slope),
            # point-wise linear convolution (no activation)
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), norm_layer(oup),
        ])            
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # use_res_connect인 경우만 connection을 연결
        # use_res_connect : stride가 1이고 input과 output의 채널 수가 같은 경우 True
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class pdcblock1(nn.Module):
    """
       기존: Conv_PDC → BN/ReLU → AvgPool(stride=2)
       변경: PDC → MobileNetV2 스타일로 대체(Stride로 다운샘플)

       반영 내용
       - 다운샘플링은 depthwise에서 stride로 수행 (MobileNet 스타일)
       - depthwise에서 채널 in=out=mid, pointwise로 outchannel을 맞춤
    """
    def __init__(self, inchannel, outchannel, expand=1.0, act='prelu', leaky_slope=0.10):
        super().__init__()
        self.block = InvertedResidual(inchannel, outchannel, stride=2, expand_ratio=expand, act=act, leaky_slope=leaky_slope)

    def forward(self, x):
        return self.block(x)


class pdcblock2(nn.Module):
    """
       기존: Conv_PDC → BN/ReLU → Conv_PDC → BN/ReLU → AvgPool(stride=2)
       변경: [PDC → MobileNetV2 스타일로 대체(Stride로 다운샘플)]

       반영 내용
       - 첫 블록은 해상도 유지(stride=1), 두 번째 블록에서 다운샘플(stride=2)
       - 첫 블록은 다운샘플(stride=2), 두 번째 블록에서 해상도 유지(stride=1)
       - 둘 다 depthwise + pointwise 구조 유지
    """
    def __init__(self, inchannel, outchannel, expand=6.0, act='prelu', leaky_slope=0.10):
        super().__init__()
        self.block = nn.Sequential(
            InvertedResidual(inchannel, outchannel, stride=2, expand_ratio=expand, act=act, leaky_slope=leaky_slope),
            InvertedResidual(outchannel, outchannel, stride=1, expand_ratio=expand, act=act, leaky_slope=leaky_slope),
        )

    def forward(self, x):
        return self.block(x)


class EnhanceMHSA_v3(nn.Module):
    def __init__(self, input_size, channels, d_k, d_v, stride, heads,
                 dropout=0.0, blocks=1, expand=4.0, use_all_stride=False, norm_layer=nn.BatchNorm2d, act='prelu', leaky_slope=0.10):
        super().__init__()
        self._printed = False

        # k / v 경로를 movilenetV2 스타일로
        def make_ir_stack():
            layers = []
            for i in range(blocks):
                s = stride if (use_all_stride or i == 0) else 1
                layers.append(InvertedResidual(channels, channels, stride=s, expand_ratio=expand, norm_layer=norm_layer, act=act, leaky_slope=leaky_slope))
            return nn.Sequential(*layers)

        self.k_backbone = make_ir_stack()
        self.v_backbone = make_ir_stack()

        # Q/K/V projection (logit-space는 동일)
        self.fc_q = nn.Linear(channels, heads * d_k)
        self.fc_k = nn.Linear(channels, heads * d_k)
        self.fc_v = nn.Linear(channels, heads * d_v)
        self.fc_o = nn.Linear(heads * d_v, channels)

        self.ln = nn.LayerNorm(channels)

        self.channels = channels
        self.d_k = d_k
        self.d_v = d_v
        self.heads = heads
        self.dropout = dropout
        self.scaled_factor = self.d_k ** -0.5

        # 다운스케일 비율 (모두에 stride를 쓴다면 stride**blocks)
        self.down_factor = (stride ** blocks) if use_all_stride else stride
        khw = (input_size // self.down_factor) ** 2
        self.B = nn.Parameter(torch.zeros(1, heads, input_size ** 2, khw), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape
        if not self._printed:
            print(f"[Before Attention] H×W = {h}×{w} (tokens={h*w})")

        # Q : 채널 정규화 후 선형
        x_reshape = x.view(b, c, h*w).permute(0, 2, 1)
        x_reshape = self.ln(x_reshape)
        q = self.fc_q(x_reshape).view(b, h*w, self.heads, self.d_k).permute(0, 2, 1, 3).contiguous()

        # K : V2 백본
        k = self.k_backbone(x)
        kb, kc, kh, kw = k.shape
        if not self._printed:
            print(f"[After IR(V2)] H×W = {kh}×{kw} (tokens={kh*kw})")
            self._printed = True
        k = k.view(kb, kc, kh*kw).permute(0, 2, 1).contiguous()
        k = self.fc_k(k).view(kb, kh*kw, self.heads, self.d_k).permute(0, 2, 1, 3).contiguous()

        # V : V2 백본
        v = self.v_backbone(x)
        vb, vc, vh, vw = v.shape
        v = v.view(vb, vc, vh*vw).permute(0, 2, 1).contiguous()
        v = self.fc_v(v).view(vb, vh*vw, self.heads, self.d_v).permute(0, 2, 1, 3).contiguous()

        # Attention
        attn = torch.einsum('... i d, ... j d -> ... i j', q, k) * self.scaled_factor
        attn = torch.softmax(attn + self.B, dim=-1)  # [b, heads, h*w, kh*kw]

        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous()
        out = self.fc_o(out.view(b, h*w, self.heads * self.d_v)).view(b, self.channels, h, w)
        return out + x


def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_str = math.floor(previous_conv_size[0] / out_pool_size[i])
        w_str = math.floor(previous_conv_size[1] / out_pool_size[i])
        #maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_str, w_str))
        maxpool = nn.AvgPool2d((h_wid, w_wid), stride=(h_str, w_str))
        x = maxpool(previous_conv)
        if(i == 0):
            spp = x.view(num_sample,-1)
            # print("spp size:",spp.size())
        else:
            # print("size:",spp.size())
            spp = torch.cat((spp,x.view(num_sample,-1)), 1)
    return spp


class Net(nn.Module):
  def __init__(self, fixed_indices, act='prelu', leaky_slope=0.10):
    super(Net, self).__init__()

    k = len(fixed_indices) # 고정 HPF 채널 수(K)
    self.group1 = HPF(bank_size=k, fixed_indices=fixed_indices)

    # 입력 채널 = 3 * K (Y/U/V 각각 K)
    self.group2 = pdcblock1(3 * k, 32, act=act, leaky_slope=leaky_slope)
    self.group3 = pdcblock2(32, 64, act=act, leaky_slope=leaky_slope)
    self.group4 = pdcblock2(64, 128, act=act, leaky_slope=leaky_slope)
    self.entrans = EnhanceMHSA_v3(64, 128, 64, 64, 2, 2, 0.0, expand=4.0, use_all_stride=False, act=act, leaky_slope=leaky_slope)

    self.fc1 = nn.Linear(128 * (4*4 + 2*2 + 1*1), 1024)
    self.fc2 = nn.Linear(1024, 2)

  def forward(self, input):
    y = self.group1(input[:,0:1,:,:])
    u = self.group1(input[:,1:2,:,:])
    v = self.group1(input[:,2:3,:,:])
    output = torch.cat([y,u,v], dim=1)
    output = self.group2(output)
    output = self.group3(output)
    output = self.group4(output)
    output = self.entrans(output)
    spp = spatial_pyramid_pool(output, output.size(0),
                               [int(output.size(2)),int(output.size(3))],
                               [4,2,1])
    output = self.fc1(spp)
    output = self.fc2(output)
    return output


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(model, device, train_loader, optimizer, epoch, args=None, scheduler=None):
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()

    model.train()
    end = time.time()

    for i, sample in enumerate(train_loader):
        data_time.update(time.time() - end)

        data, label = sample['data'], sample['label']
        shape = list(data.size())                 # (B, 2, C, H, W)
        data  = data.reshape(shape[0] * shape[1], *shape[2:])  # (2B, C, H, W)
        label = label.reshape(-1)                 # (2B,)

        data, label = data.to(device), label.to(device)

        optimizer.zero_grad(set_to_none=True)
        end = time.time()

        # forward
        logits = model(data)

        # --- CE term ---
        ce_term = F.cross_entropy(logits, label, reduction='mean')

        loss = ce_term

        # backward
        losses.update(loss.item(), data.size(0))
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % TRAIN_PRINT_FREQUENCY == 0:
            cur_lr = optimizer.param_groups[0]['lr']
            logging.info(
                f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                f'LR {cur_lr:.3e}\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'CE {ce_term.item():.4f}'
            )


def adjust_bn_stats(model, device, train_loader):
    model.train()

    with torch.no_grad():
        for sample in train_loader:
            data, label = sample['data'], sample['label']

            shape = list(data.size())
            data = data.reshape(shape[0] * shape[1], *shape[2:])
            label = label.reshape(-1)

            data, label = data.to(device), label.to(device)

            output = model(data)


def evaluate(model, device, eval_loader, epoch, optimizer, best_acc, PARAMS_PATH, PARAMS_PATH1, TMP):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for sample in eval_loader:
            data, label = sample['data'], sample['label']

            shape = list(data.size())
            data = data.reshape(shape[0] * shape[1], *shape[2:])
            label = label.reshape(-1)

            data, label = data.to(device), label.to(device)

            output = model(data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).sum().item()

    accuracy = correct / (len(eval_loader.dataset) * 2)

    all_state = {
        'original_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'epoch': epoch
      }
    torch.save(all_state, PARAMS_PATH1)

    if accuracy > best_acc and epoch > TMP:
        best_acc = accuracy
        all_state = {
            'original_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(all_state, PARAMS_PATH)

    logging.info('-' * 8)
    logging.info('Eval accuracy: {:.4f}'.format(accuracy))
    logging.info('Best accuracy:{:.4f}'.format(best_acc))
    logging.info('-' * 8)
    return best_acc


def initWeights(module, nonlinearity='relu', a=0.0): # 신경망의 가중치 초기화(weight initialization) 를 해주는 함수
    if type(module) == nn.Conv2d:
        if module.weight.requires_grad:
            nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity=nonlinearity, a=a)
            # mode='fan_in'은 입력 채널 수(fan_in)를 기준으로 분산을 맞추겠다는 뜻

    if type(module) == nn.Linear:
        nn.init.normal_(module.weight.data, mean=0, std=0.01)


class AugData():
    def __call__(self, sample):
        data, label = sample['data'], sample['label']
        name = sample.get('name') if isinstance(sample, dict) else None

        rot = random.randint(0, 3)

        data = np.rot90(data, rot, axes=[2, 3]).copy()  

        if random.random() < 0.5:
            data = np.flip(data, axis=2).copy()

        new_sample = {'data': data, 'label': label}
        if name is not None:
            new_sample['name'] = name

        return new_sample


class ToTensor():
    def __call__(self, sample):
        data, label = sample['data'], sample['label']
        name = sample.get('name') if isinstance(sample, dict) else None

        data = data.astype(np.float32)

        new_sample = {
            'data': torch.from_numpy(data),
            'label': torch.from_numpy(label).long(),
        }
        if name is not None:
            new_sample['name'] = name

        return new_sample


class MyDataset(Dataset):
    def __init__(self, index_path, BOSSBASE_COVER_DIR, BOSSBASE_STEGO_DIR, transform=None):
        self.index_list = np.load(index_path)
        self.transform = transform

        self.bossbase_cover_path = os.path.join(BOSSBASE_COVER_DIR, '{}.jpg')
        self.bossbase_stego_path = os.path.join(BOSSBASE_STEGO_DIR, '{}.jpg')
        # self.bossbase_cover_path = os.path.join(BOSSBASE_COVER_DIR, '{}.png')
        # self.bossbase_stego_path = os.path.join(BOSSBASE_STEGO_DIR, '{}.png')

    def __len__(self):
        return self.index_list.shape[0]

    def __getitem__(self, idx):
        file_index = self.index_list[idx]

        cover_path = self.bossbase_cover_path.format(file_index)
        stego_path = self.bossbase_stego_path.format(file_index)

        cover_data = cv2.imread(cover_path, -1)
        # cover_data = cv2.cvtColor(cover_data, cv2.COLOR_BGR2RGB)
        cover_data = cv2.cvtColor(cover_data, cv2.COLOR_BGR2YCrCb)
        cover_data = np.transpose(cover_data, (2, 0, 1))
        stego_data = cv2.imread(stego_path, -1)
        # stego_data = cv2.cvtColor(stego_data, cv2.COLOR_BGR2RGB)
        stego_data = cv2.cvtColor(stego_data, cv2.COLOR_BGR2YCrCb)
        stego_data = np.transpose(stego_data, (2, 0, 1))

        data = np.stack([cover_data, stego_data])
        label = np.array([0, 1], dtype='int32')

        sample = {'data': data, 'label': label, 'name': file_index}

        if self.transform:
            sample = self.transform(sample)

        return sample


def setLogger(log_path, mode='a'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path, mode=mode)
        file_handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s', '%Y-%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def main(args):

    statePath = args.statePath

    device = torch.device("cuda")

    kwargs = {'num_workers': 1, 'pin_memory': True}

    train_transform = transforms.Compose([
        AugData(),
        ToTensor()
    ])

    eval_transform = transforms.Compose([
        ToTensor()
    ])

    DATASET_INDEX = args.DATASET_INDEX
    STEGANOGRAPHY = args.STEGANOGRAPHY
    EMBEDDING_RATE = args.EMBEDDING_RATE
    TIMES = args.times

    COVER_DIR = r'Alaska2/QF_90/cover'
    STEGO_DIR = r'Alaska2/QF_90/Stego/conseal_nsF5_0.4'

    TEST_COVER = COVER_DIR
    TEST_STEGO  = STEGO_DIR

    TRAIN_INDEX_PATH = f'index_list/train_index_{DATASET_INDEX}.npy'
    VALID_INDEX_PATH = f'index_list/valid_index_{DATASET_INDEX}.npy'
    TEST_INDEX_PATH  = f'index_list/var_test_index_{DATASET_INDEX}.npy'

    # LOAD_RATE = float(EMBEDDING_RATE) + 0.1
    # LOAD_RATE = round(LOAD_RATE, 1)

    PARAMS_NAME = '{}-{}-{}-params-{}-lr={}.pt'.format(STEGANOGRAPHY, EMBEDDING_RATE, DATASET_INDEX, TIMES, args.lr)
    PARAMS_NAME1 = '{}-{}-{}-process-params-{}-lr={}.pt'.format(STEGANOGRAPHY, EMBEDDING_RATE, DATASET_INDEX, TIMES, args.lr)
    LOG_NAME = '{}-{}-{}-model_log-{}-lr={}.log'.format(STEGANOGRAPHY, EMBEDDING_RATE, DATASET_INDEX, TIMES, args.lr)
    #statePath='./log/'+PARAMS_NAME1

    PARAMS_PATH = os.path.join(OUTPUT_PATH, PARAMS_NAME)
    PARAMS_PATH1 = os.path.join(OUTPUT_PATH, PARAMS_NAME1)
    LOG_PATH = os.path.join(OUTPUT_PATH, LOG_NAME)

    Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)
    setLogger(LOG_PATH, mode='w')


    train_dataset = MyDataset(TRAIN_INDEX_PATH, COVER_DIR, STEGO_DIR, train_transform)
    valid_dataset = MyDataset(VALID_INDEX_PATH, COVER_DIR, STEGO_DIR, eval_transform)
    test_dataset = MyDataset(TEST_INDEX_PATH, TEST_COVER, TEST_STEGO, eval_transform)

    batch_size = args.batch_size

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, **kwargs)

    # ── HPF 선택은 "증강 없는" 로더로 수행 ──
    measure_dataset = MyDataset(TRAIN_INDEX_PATH, COVER_DIR, STEGO_DIR, transform=eval_transform)  # AugData 제거
    measure_loader  = DataLoader(measure_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    # print("[HPF] measuring mean energy over train set...")
    # fixed_indices = select_fixed_indices(loader=measure_loader, device=device, bank_size=args.hpf_bank, total_k=args.hpf_topk, gabor_k=None)
    if args.hpf_mode == 'random':
        k = args.hpf_random_topk if args.hpf_random_topk is not None else args.hpf_topk
        k = max(0, min(k, args.hpf_bank))
        fixed_indices = random.sample(range(args.hpf_bank), k)
        print(f"[HPF] randomly picked {k} filters out of {args.hpf_bank}: {fixed_indices}")
    else:
        print("[HPF] learning gate to select top-K filters...")
        fixed_indices = learnable_select_indices(loader=measure_loader, device=device, bank_size=args.hpf_bank, top_k=args.hpf_topk, epochs=args.hpf_learn_epochs, gabor_k=args.hpf_gabor)
    k_fixed = len(fixed_indices)
    print(f"[HPF] fixed K = {k_fixed}  indices = {fixed_indices}")

    """
    가중치 초기화 순서
    - model.apply(initWeights) 는 DataParallel 감싸기 전에 호출해야 각 submodule에 올바르게 적용됨
    """
    # act에 맞춰 Kaiming gain 설정
    nl, a = 'relu', 0.0
    if args.act in ('prelu', 'leakyrelu'):
        nl = 'leaky_relu'
        a  = (0.25 if args.act == 'prelu' else args.leaky_slope)
    model = Net(fixed_indices=fixed_indices, act=args.act, leaky_slope=args.leaky_slope).to(device)
    model.apply(lambda m: initWeights(m, nonlinearity=nl, a=a))
    model = torch.nn.DataParallel(model) # apply는 인자를 직접 못 주니까, lambda로 감싸서 넘김
    model = model.cuda()
    model.eval()

    # ── FLOPs 계산 & 출력 ──
    # (1) 전체 모델 이론 FLOPs
    dummy = torch.randn(1, 3, 512, 512).to(device)
    flops_all = FlopCountAnalysis(model.module, (dummy,)).total()

    # (2) 어텐션 모듈만 따로
    attn = model.module.entrans
    dummy_attn = torch.randn(1, 128, 64, 64).to(device) # (B, C, H, W) = EnhanceMHSA 입력 크기
    flops_attn = FlopCountAnalysis(attn, (dummy_attn,)).total()

    # (3) 실제 벤치마크 GFLOPS
    inputs = torch.randn(batch_size, 3, 512, 512).to(device)
    # 워밍업
    with torch.no_grad():
        for _ in range(10):
            _ = model(inputs)
    torch.cuda.synchronize()
    # 측정
    starter, ender = torch.cuda.Event(True), torch.cuda.Event(True)
    with torch.no_grad():
        starter.record()
        for _ in range(50):
            _ = model(inputs)
        ender.record()
    torch.cuda.synchronize()
    elapsed_ms = starter.elapsed_time(ender) / 50
    elapsed_s = elapsed_ms / 1000
    gflops = (flops_all * batch_size) / elapsed_s / 1e9
    print(f"Per-sample FLOPs: {flops_all:.2e}")
    print(f"어텐션 모듈 FLOPs: {flops_attn:.2e}")
    print(f"어텐션 비중: {flops_attn/flops_all*100:.1f}%")
    print(f"Batch size: {batch_size}")
    print(f"Elapsed per batch: {elapsed_ms:.1f} ms")
    print(f"Effective GFLOPS: {gflops:.1f}")

    """
    Optimizer: AdamW로 변경 + 모든 하이퍼파라미터(args) 외부화
    """
    optimizer = build_optimizer(model, args)

    EPOCHS = args.epochs
    steps_per_epoch = len(train_loader)
    total_iters = steps_per_epoch * EPOCHS

    if statePath:
        logging.info('-' * 8)
        logging.info('Load state_dict in {}'.format(statePath))
        logging.info('-' * 8)

        all_state = torch.load(statePath)

        original_state = all_state['original_state']
        optimizer_state = all_state['optimizer_state']
        epoch = all_state['epoch']

        model.load_state_dict(original_state)
        optimizer.load_state_dict(optimizer_state)

        startEpoch = epoch + 1

    else:
        startEpoch = 1


    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=DECAY_EPOCH, gamma=0.1)
    """
    Scheduler: CosineAnnealingLR (iteration 단위)
    - T_max = total_iters (steps_per_epoch × epochs)
    """
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters, eta_min=1e-5)
    best_acc = 0.0
    for epoch in range(startEpoch, EPOCHS + 1):
        train(model, device, train_loader, optimizer, epoch, args, scheduler)

        """
        BN 통계 재적산 토글 (args.bn_recalc)
        - 평가 직전 train 모드로 돌려 미니배치에 맞게 BN running stats를 업데이트
        - 켜고/끄고 성능 차이 비교 가능
        """
        if epoch % EVAL_PRINT_FREQUENCY == 0:
            if args.bn_recalc:
                adjust_bn_stats(model, device, train_loader)
            best_acc = evaluate(model, device, valid_loader, epoch, optimizer,
                                best_acc, PARAMS_PATH, PARAMS_PATH1, args.save_after)

    logging.info('\nTest set accuracy: \n')

    all_state = torch.load(PARAMS_PATH)
    original_state = all_state['original_state']
    optimizer_state = all_state['optimizer_state']
    model.load_state_dict(original_state)
    optimizer.load_state_dict(optimizer_state)

    if args.bn_recalc:
        adjust_bn_stats(model, device, train_loader)

    # 테스트는 저장 금지 (TMP를 매우 크게, best_acc도 크게 전달)
    _ = evaluate(model, device, test_loader, epoch, optimizer,
                best_acc=best_acc, PARAMS_PATH=PARAMS_PATH, PARAMS_PATH1=PARAMS_PATH1, TMP=10**9)


def myParseArgs():
    parser = argparse.ArgumentParser()

    # 데이터/경로 태그
    parser.add_argument(
        '-i', 
        '--DATASET_INDEX', 
        type=str, default='1',
        help='which split index to load (affects index_list/* files)'
    )

    parser.add_argument(
        '-alg', 
        '--STEGANOGRAPHY', 
        type=str, 
        required=True,
        choices=['HILL-CMDC', 'SUNIWARD-CMDC','nsf5_0.5', 'nsf5_0.4', 'nsf5_0.2',
                'J-UNIWARD_0.2', 'J-UNIWARD_0.4',
                'UERD_0.2', 'UERD_0.4',
                'LSB_0.2', 'LSB_0.4',],
        help='experiment tag for filenames'
    )

    parser.add_argument(
        '-rate',
        '--EMBEDDING_RATE',
        type=str, 
        required=False,
        choices=['0.2','0.3','0.4'],
        help='another tag for filenames (not used in training logic)'
    )

    parser.add_argument(
        '-t', 
        '--times', 
        type=str,
        default='',
        help='run suffix tag for filenames'
    )

    # ── 디바이스/체크포인트 ──
    parser.add_argument(
        '-g', 
        '--gpuNum', 
        type=str,required=True,
        choices=[str(i) for i in range(8)],
        help='which GPU to use (CUDA_VISIBLE_DEVICES)'
    )

    parser.add_argument(
        '-l',
        '--statePath',
        type=str,
        default='',
        help='path to resume checkpoint'
    )

    # 학습 크기/일정
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=8
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=50
    )

    parser.add_argument(
        '--save-after', 
        type=int, 
        default=25,
        help='start saving best model after this epoch'
    )

    # BN 러닝스탯 재계산
    parser.add_argument(
        '--no-bn-recalc', 
        dest='bn_recalc', # args.bn_recalc 라는 이름의 변수에 담기
        action='store_false', #  기본은 bn_recalc 값을 True 로 저장
        help='disable BN running-stats recalculation before eval'
    )

    parser.set_defaults(
        bn_recalc=True
    )

    # 옵티마이저/정규화
    parser.add_argument(
        '--optimizer', 
        type=str, 
        default='adamw',
        choices=['sgd','adamw']
    )

    parser.add_argument(
        '--lr', 
        type=float, 
        default=5e-4
    )
    
    parser.add_argument(
        '--weight-decay', 
        type=float, 
        default=1e-2
    )

    parser.add_argument(
        '--momentum', 
        type=float,
        default=0.9,
        help='used only optimizer=sgd'
    )

    # ── HPF 선택 ──
    # myParseArgs() 안에 추가
    parser.add_argument(
        '--hpf-mode',
        type=str,
        default='learn',
        choices=['learn', 'random'],
        help='HPF selection strategy: learn = gate-based top-K, random = sample filters without gate training'
    )

    parser.add_argument(
        '--hpf-topk', 
        type=int, 
        default=31,
        help='use fixed HPF numbers'
    )

    parser.add_argument(
        '--hpf-random-topk',
        type=int,
        default=None,
        help='when --hpf-mode=random, sample this many filters (defaults to hpf-topk if not set)'
    )
    
    parser.add_argument(
        '--hpf-bank', 
        type=int, 
        default=62,
        help='HPF bank size (62=all SRM+Gabor). Usually keep 62.'
    )

    parser.add_argument(
        '--hpf-gabor', 
        type=int, 
        default=None,
        help='force of Gabor filters (default ceil(total*32/62))'
    )

    parser.add_argument(
        '--hpf-learn-epochs', 
        type=int, 
        default=3
    )

    # 옵티마이저 옵션들 다음에 추가
    parser.add_argument(
        '--act', 
        type=str, 
        default='prelu',
        choices=['prelu','leakyrelu','relu6','relu','silu'],
        help='activation for conv blocks (prelu/leakyrelu/relu6/relu/silu)'
    )

    parser.add_argument(
        '--leaky-slope', 
        type=float, 
        default=0.10,
        help='negative slope for LeakyReLU when --act=leakyrelu'
    )


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = myParseArgs()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuNum
    main(args)
