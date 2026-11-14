import torch
import torch.nn as nn
from openstl.modules.rdmn_modules import (
    RDMNDifferentialCell,
    RDMNSpatioTemporalCell,
)
from openstl.modules.mim_modules import MIMN
from openstl.modules.rdmn_motion_guided import MotionGuided


class RDMNModel(nn.Module):
    """Recurrent Differential Memory Network for frame prediction."""

    def __init__(self, num_layers, num_hidden, att_hidden, configs):
        super().__init__()
        T, C, H, W = configs.in_shape

        self.configs = configs
        height = H // configs.patch_size
        width = W // configs.patch_size

        self.patch_ch = C * (configs.patch_size**2)
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.neighbour = 3
        self.motion_hidden = 2 * self.neighbour * self.neighbour
        self.att_hidden = att_hidden
        self.MSE_criterion = nn.MSELoss()

        cell_list = []
        # 第 0 层为带 SGFM 的 ST-LSTM，第 1..n-1 层为带 SGFM 的 MIM 风格单元
        for i in range(num_layers):
            in_channel = self.patch_ch if i == 0 else num_hidden[i - 1]
            if i == 0:
                cell_list.append(
                    RDMNSpatioTemporalCell(
                        in_channel, num_hidden[i], self.att_hidden, height,
                        width, configs.filter_size, configs.stride,
                        configs.layer_norm), )
            else:
                cell_list.append(
                    RDMNDifferentialCell(in_channel, num_hidden[i],
                                         self.att_hidden, height, width,
                                         configs.filter_size, configs.stride,
                                         configs.layer_norm), )

        mim_diff_list = []
        # 是 MIMN 层，用于将隐藏状态的差分转换为细胞状态的差分。0~N-1层需要差分，N层不需要差分。
        for i in range(num_layers - 1):
            mim_diff_list.append(
                MIMN(num_hidden[i], num_hidden[i + 1], height, width,
                     configs.filter_size, configs.stride,
                     configs.layer_norm), )
        self.mim_diff_list = nn.ModuleList(mim_diff_list)

        enc_list = []
        # 是卷积层，用于将隐藏状态降维，接在第 0 层之后，用于后续的 motion_list 和 dec_list。
        for i in range(num_layers - 1):
            enc_list.append(
                nn.Conv2d(
                    num_hidden[i],
                    num_hidden[i] // 4,
                    kernel_size=configs.filter_size,
                    stride=2,
                    padding=configs.filter_size // 2), )
        motion_list = []
        # 是 MotionGuided 层，用于将隐藏状态的偏移量转换为运动高速公路。
        for i in range(num_layers - 1):
            motion_list.append(
                MotionGuided(num_hidden[i] // 4, self.motion_hidden,
                             self.neighbour))
        dec_list = []
        # 是卷积层，用于将隐藏状态升维，接在 motion_list 之后，用于后续的 gate_list。
        for i in range(num_layers - 1):
            dec_list.append(
                nn.ConvTranspose2d(
                    num_hidden[i] // 4,
                    num_hidden[i],
                    kernel_size=4,
                    stride=2,
                    padding=1), )
        gate_list = []
        # 是卷积层，用于将隐藏状态和卷积后的隐藏状态拼接后，再进行卷积，得到门控信号。
        for i in range(num_layers - 1):
            gate_list.append(
                nn.Conv2d(
                    num_hidden[i] * 2,
                    num_hidden[i],
                    kernel_size=configs.filter_size,
                    stride=1,
                    padding=configs.filter_size // 2), )
        self.gate_list = nn.ModuleList(gate_list)
        self.cell_list = nn.ModuleList(cell_list)
        self.motion_list = nn.ModuleList(motion_list)
        self.enc_list = nn.ModuleList(enc_list)
        self.dec_list = nn.ModuleList(dec_list)
        # 是卷积层，用于将隐藏状态升维，接在 dec_list 之后，用于后续的 conv_last。
        self.conv_last = nn.Conv2d(
            num_hidden[num_layers - 1],
            self.patch_ch,
            1,
            stride=1,
            padding=0,
            bias=False)
        self.conv_first_v = nn.Conv2d(
            self.patch_ch, num_hidden[0], 1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true, **kwargs):
        device = frames_tensor.device
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        next_frames = []
        # 状态初始化
        # h_t: 隐藏状态
        # c_t: 细胞状态
        # h_t_conv: 卷积后的隐藏状态
        # h_t_conv_offset: 卷积后的隐藏状态的偏移
        # mean: 均值
        h_t = []
        c_t = []
        h_t_conv = []
        h_t_conv_offset = []
        mean = []

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        # 第 0 层
        for i in range(self.num_layers):
            zeros = torch.empty([batch, self.num_hidden[i], height, width],
                                device=device)
            nn.init.xavier_normal_(zeros)
            h_t.append(zeros)
            c_t.append(zeros)
        # 第 1..n-1 层
        for i in range(self.num_layers - 1):
            zeros = torch.empty(
                [batch, self.num_hidden[i] // 4, height // 2, width // 2],
                device=device)
            nn.init.xavier_normal_(zeros)
            h_t_conv.append(zeros)
            zeros = torch.empty(
                [batch, self.motion_hidden, height // 2, width // 2],
                device=device)
            nn.init.xavier_normal_(zeros)
            h_t_conv_offset.append(zeros)
            mean.append(zeros)
        # mem: 记忆
        # motion_highway:
        mem = torch.empty([batch, self.num_hidden[0], height, width],
                          device=device)
        motion_highway = torch.empty(
            [batch, self.num_hidden[0], height, width], device=device)
        nn.init.xavier_normal_(mem)
        nn.init.xavier_normal_(motion_highway)

        # hidden_state_diff: 隐藏状态的差分
        # cell_state_diff: 细胞状态的差分
        hidden_state_diff = [None] * (self.num_layers - 1)
        cell_state_diff = [None] * (self.num_layers - 1)
        # x_gen: 生成的帧
        x_gen = None

        # 大循环：遍历所有时间步
        for t in range(self.configs.pre_seq_length +
                       self.configs.aft_seq_length - 1):
            # 输入帧
            if t < self.configs.pre_seq_length:
                net = frames[:, t]
            else:
                if x_gen is None:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.configs.pre_seq_length] * frames[:, t] + \
                          (1 - mask_true[:, t - self.configs.pre_seq_length]) * x_gen

            # 第 0 层
            # preh: 上一时间步的隐藏状态
            # motion_highway: 运动高速公路
            # h_t[0]: 隐藏状态
            # c_t[0]: 细胞状态
            # mem: 记忆
            # motion_highway: 运动高速公路
            preh = h_t[0].clone()
            # 卷积第一层
            motion_highway = self.conv_first_v(net)
            # 第 0 层
            # net: 输入帧
            h_t[0], c_t[0], mem, motion_highway = self.cell_list[0](
                net, h_t[0], c_t[0], mem, motion_highway)
            net = self.enc_list[0](h_t[0])
            h_t_conv[0], h_t_conv_offset[0], mean[0] = self.motion_list[0](
                net, h_t_conv_offset[0], mean[0])
            h_t_tmp = self.dec_list[0](h_t_conv[0])
            o_t = torch.sigmoid(self.gate_list[0](torch.cat([h_t_tmp, h_t[0]],
                                                            dim=1)))
            h_t[0] = o_t * h_t_tmp + (1 - o_t) * h_t[0]

            if t > 0:
                hidden_state_diff[0], cell_state_diff[0] = self.mim_diff_list[
                    0](h_t[0] - preh, hidden_state_diff[0], cell_state_diff[0])
                for i in range(1, self.num_layers - 1):
                    hidden_state_diff[i], cell_state_diff[
                        i] = self.mim_diff_list[i](hidden_state_diff[i - 1],
                                                   hidden_state_diff[i],
                                                   cell_state_diff[i])
            else:
                for i in range(self.num_layers - 1):
                    _ = self.mim_diff_list[i](torch.zeros_like(h_t[i]), None,
                                              None)
            # 第 1..n-1 层
            for i in range(1, self.num_layers - 1):
                diff_h = hidden_state_diff[i -
                                           1] if t > 0 else torch.zeros_like(
                                               h_t[i])
                # 比第0层，增加输入为 diff_h: 隐藏状态的差分
                h_t[i], c_t[i], mem, motion_highway = self.cell_list[i](
                    h_t[i - 1], diff_h, h_t[i], c_t[i], mem, motion_highway)
                net = self.enc_list[i](h_t[i])
                h_t_conv[i], h_t_conv_offset[i], mean[i] = self.motion_list[i](
                    net, h_t_conv_offset[i], mean[i])
                h_t_tmp = self.dec_list[i](h_t_conv[i])
                o_t = torch.sigmoid(self.gate_list[i](torch.cat(
                    [h_t_tmp, h_t[i]], dim=1)))
                h_t[i] = o_t * h_t_tmp + (1 - o_t) * h_t[i]
            # 第 N 层
            diff_h_last = hidden_state_diff[self.num_layers -
                                            2] if t > 0 else torch.zeros_like(
                                                h_t[self.num_layers - 1])
            h_t[self.num_layers - 1], c_t[
                self.num_layers -
                1], mem, motion_highway = self.cell_list[self.num_layers - 1](
                    h_t[self.num_layers - 2], diff_h_last,
                    h_t[self.num_layers - 1], c_t[self.num_layers - 1], mem,
                    motion_highway)
            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        next_frames = torch.stack(
            next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        if kwargs.get('return_loss', True):
            loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        else:
            loss = None
        return next_frames, loss
