import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiScaleConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes=[3, 5, 7, 11],
                 stride=1,
                 padding='same'):
        super(MultiScaleConv, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=k,
                stride=stride,
                padding=k // 2) for k in kernel_sizes
        ])

    def forward(self, x):
        return sum(conv(x) for conv in self.convs)


class SelfAttentionMemory(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # h(hidden): layer q, k, v
        # m(memory): layer k2, v2
        # layer z, m are for layer after concat(attention_h, attention_m)

        # layer_q, k, v are for h (hidden) layer
        # Layer_ k2, v2 are for m (memory) layer
        # Layer_z, m are using after concatinating attention_h and attention_m layer

        self.layer_q = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_k = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_k2 = nn.Conv2d(input_dim, hidden_dim, 1)
        self.layer_v = nn.Conv2d(input_dim, input_dim, 1)
        self.layer_v2 = nn.Conv2d(input_dim, input_dim, 1)

        self.layer_z = nn.Conv2d(input_dim * 2, input_dim * 2, 1)
        self.layer_m = nn.Conv2d(input_dim * 3, input_dim * 3, 1)
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

    def forward(self, h, m):
        batch_size, channel, H, W = h.shape
        # feature aggregation
        ##### hidden h attention #####
        K_h = self.layer_k(h)
        Q_h = self.layer_q(h)
        K_h = K_h.view(batch_size, self.hidden_dim, H * W)
        Q_h = Q_h.view(batch_size, self.hidden_dim, H * W)
        Q_h = Q_h.transpose(1, 2)

        A_h = torch.softmax(
            torch.bmm(Q_h, K_h), dim=-1)  # batch_size, H*W, H*W

        V_h = self.layer_v(h)
        V_h = V_h.view(batch_size, self.input_dim, H * W)
        Z_h = torch.matmul(A_h, V_h.permute(0, 2, 1))

        ###### memory m attention #####
        K_m = self.layer_k2(m)
        V_m = self.layer_v2(m)
        K_m = K_m.view(batch_size, self.hidden_dim, H * W)
        V_m = V_m.view(batch_size, self.input_dim, H * W)
        A_m = torch.softmax(torch.bmm(Q_h, K_m), dim=-1)
        V_m = self.layer_v2(m)
        V_m = V_m.view(batch_size, self.input_dim, H * W)
        Z_m = torch.matmul(A_m, V_m.permute(0, 2, 1))
        Z_h = Z_h.transpose(1, 2).view(batch_size, self.input_dim, H, W)
        Z_m = Z_m.transpose(1, 2).view(batch_size, self.input_dim, H, W)

        ### Z_h & Z_m (from attention) then, concat then computation ####
        W_z = torch.cat([Z_h, Z_m], dim=1)
        Z = self.layer_z(W_z)
        ## Memory Updating (Ref: SA-ConvLSTM)
        combined = self.layer_m(torch.cat([Z, h], dim=1))  # 3 * input_dim
        mo, mg, mi = torch.split(combined, self.input_dim, dim=1)
        ### (Ref: SA-ConvLSTM)
        mi = torch.sigmoid(mi)
        new_m = (1 - mi) * m + mi * torch.tanh(mg)
        new_m = new_m
        new_h = torch.sigmoid(mo) * new_m

        return new_h, new_m


class SpectralGlobalFocus(nn.Module):

    def __init__(self, in_channels, num_hidden, height, width):
        """
        基于频域分析的全局聚焦模块 (S-GFM)
        Args:
            in_channels: 输入特征的通道数 (即 num_hidden)
            height: 特征图的高度
            width: 特征图的宽度
        注意: FFT 操作对输入尺寸敏感，训练和测试时输入分辨率必须保持一致，或在 forward 中动态调整。
        """
        super().__init__()

        self.h = height
        self.w = width
        # rfft2 (实数到复数) 会将宽度维度压缩为 W/2 + 1
        self.freq_w = width // 2 + 1

        # 1. 【新增】特征映射层 (Input Projection)
        # 将原始图像 (1通道) 映射到 特征空间 (hidden_dim 通道)
        # 这一步对应原 GFM 中的 self.conv
        self.input_proj = nn.Sequential(
            # 注意：使用了 BN 后，Conv 的 bias 必须设为 False，因为 BN 自带 bias，设为 True 会浪费参数且不稳
            nn.Conv2d(
                in_channels, num_hidden, kernel_size=3, padding=1, bias=False),
            # 它能让进入 FFT 的特征分布更标准，加速收敛
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(inplace=True))
        # 2. 定义可学习的复数权重 (Spectral Weights)
        # 这是一个在频域的全局滤波器。
        # 形状: [Channel, Height, Width_Freq, 2] (最后一位 2 代表实部和虚部)
        # 注意：通道数应该是 num_hidden，因为 input_proj 的输出是 num_hidden 通道
        # 初始化: 使用较小的方差，避免初始阶段破坏特征分布
        scale = 0.02
        self.complex_weight = nn.Parameter(scale * torch.randn(
            num_hidden, self.h, self.freq_w, 2, dtype=torch.float32))

        # 3. 后处理层 (Post-processing)
        # 将频域提取的全局特征与通道间信息进行融合，并引入非线性
        # 注意：输入输出通道数都应该是 num_hidden
        self.post_conv = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=1),
            nn.ReLU(inplace=True)  # 或者使用 LeakyReLU
        )

    def forward(self, x):
        """
        x: [Batch, Channel, Height, Width]
        """
        B, C, H, W = x.shape

        # --- 步骤 1: 升维提取特征 ---
        x_feat = self.input_proj(x)
        # x_feat shape: [B, hidden_dim, H, W] (例如 [16, 64, 64, 64])

        # --- 步骤 2: 变换到频域 ---
        # 【重要】cuFFT 在 half precision (fp16) 模式下只支持尺寸是 2 的幂次方的输入
        # 如果输入尺寸不是 2 的幂次方（如 48x48），需要临时转换为 float32
        # 保存原始数据类型
        original_dtype = x_feat.dtype

        # 如果使用 half precision 且尺寸不是 2 的幂次方，转换为 float32
        if original_dtype == torch.float16 and (H & (H - 1) != 0 or W &
                                                (W - 1) != 0):
            x_feat = x_feat.float()
            weight_dtype = torch.float32
        else:
            weight_dtype = original_dtype

        # rfft2: Real-to-Complex 2D FFT
        # norm='ortho' 保证变换前后的能量守恒
        x_freq = torch.fft.rfft2(
            x_feat, norm='ortho')  # Output: [B, C, H, W//2+1] (Complex64)

        # --- 步骤 3: 频域滤波 (全局交互) ---
        # 将参数视作复数张量，并确保数据类型匹配
        weight = torch.view_as_complex(self.complex_weight)
        if weight.dtype != x_freq.dtype:
            weight = weight.to(x_freq.dtype)

        # 核心操作: 频域点乘 (Spectral Gating)
        # 根据卷积定理，频域的点乘等价于空域的全局循环卷积
        x_freq_processed = x_freq * weight

        # --- 步骤 4: 变换回空域 ---
        # irfft2: Complex-to-Real 2D Inverse FFT
        # 必须指定输出尺寸 s=(H, W)，防止奇偶尺寸歧义
        x_spatial = torch.fft.irfft2(x_freq_processed, s=(H, W), norm='ortho')

        # 转换回原始数据类型（如果之前转换了）
        if original_dtype == torch.float16 and x_spatial.dtype == torch.float32:
            x_spatial = x_spatial.to(original_dtype)

        # --- 步骤 5: 特征融合与非线性激活 ---
        out = self.post_conv(x_spatial)
        # [B, C,H,W] -> [B, C,H,W]

        return out


class RDMNSpatioTemporalCell(nn.Module):
    """First RDMN layer with spectral focus and spatiotemporal memory."""

    def __init__(self, in_channel, num_hidden, att_hidden, height, width,
                 filter_size, stride, layer_norm):
        super().__init__()
        self.num_hidden = num_hidden
        self.att_hidden = att_hidden
        self._forget_bias = 1.0
        padding = filter_size // 2

        self.attention_layer = SelfAttentionMemory(
            num_hidden, att_hidden)
        self.global_feature_extractor = SpectralGlobalFocus(
            in_channel, num_hidden, height, width)
        self.gate = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=3, padding=1),
            nn.Sigmoid())

        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(
                    in_channel,
                    num_hidden * 7,
                    filter_size,
                    stride,
                    padding,
                    bias=False), nn.LayerNorm([num_hidden * 7, height, width]))
            self.conv_h = nn.Sequential(
                nn.Conv2d(
                    num_hidden,
                    num_hidden * 4,
                    filter_size,
                    stride,
                    padding,
                    bias=False), nn.LayerNorm([num_hidden * 4, height, width]))
            self.conv_m = nn.Sequential(
                nn.Conv2d(
                    num_hidden,
                    num_hidden * 3,
                    filter_size,
                    stride,
                    padding,
                    bias=False), nn.LayerNorm([num_hidden * 3, height, width]))
            self.conv_o = nn.Sequential(
                nn.Conv2d(
                    num_hidden * 2,
                    num_hidden,
                    filter_size,
                    stride,
                    padding,
                    bias=False), nn.LayerNorm([num_hidden, height, width]))
            self.conv_g = nn.Sequential(
                # nn.Conv2d(in_channel, num_hidden * 1, filter_size, stride, padding, bias=False),
                MultiScaleConv(
                    in_channel, num_hidden, kernel_sizes=[1, 3, 5], stride=1),
                nn.LayerNorm([num_hidden * 1, height, width]))
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(
                    in_channel,
                    num_hidden * 7,
                    filter_size,
                    stride,
                    padding,
                    bias=False), )
            self.conv_h = nn.Sequential(
                nn.Conv2d(
                    num_hidden,
                    num_hidden * 4,
                    filter_size,
                    stride,
                    padding,
                    bias=False), )
            self.conv_m = nn.Sequential(
                nn.Conv2d(
                    num_hidden,
                    num_hidden * 3,
                    filter_size,
                    stride,
                    padding,
                    bias=False), )
            self.conv_o = nn.Sequential(
                nn.Conv2d(
                    num_hidden * 2,
                    num_hidden,
                    filter_size,
                    stride,
                    padding,
                    bias=False), )
            self.conv_g = nn.Sequential(
                # nn.Conv2d(in_channel, num_hidden * 1, filter_size, stride, padding, bias=False),
                MultiScaleConv(
                    in_channel, num_hidden, kernel_sizes=[1, 3, 5],
                    stride=1), )

        self.conv_last = nn.Conv2d(
            num_hidden * 2,
            num_hidden,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)

    def forward(self, x_t, h_t, c_t, m_t, motion_highway):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        g_concat = self.conv_g(x_t)

        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(
            x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)
        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)
        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        m_new_new = self.conv_last(mem)

        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(m_new_new)

        # S-GFM 输出 [B, num_hidden, H, W]，直接用于门控
        global_features = self.global_feature_extractor(x_t)
        gate = self.gate(torch.cat([h_new, global_features], dim=1))
        # h_new = h_new * gate + global_features * (1 - gate) + h_new

        h_new = h_new * gate + global_features * (1 - gate)
        h_new = h_new * g_concat

        # Attention layer
        h_new, m_new = self.attention_layer(h_new, m_new)

        h_new = h_new + (1 - o_t) * motion_highway
        motion_highway = h_new

        return h_new, c_new, m_new, motion_highway


class RDMNDifferentialCell(nn.Module):
    """Higher RDMN layer with differential memory and spectral focus."""

    def __init__(self, in_channel, num_hidden, att_hidden, height, width,
                 filter_size, stride, layer_norm):
        super().__init__()
        self.convlstm_c = None
        self.num_hidden = num_hidden
        self.att_hidden = att_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0

        self.ct_weight = nn.Parameter(
            torch.zeros(num_hidden * 2, height, width))
        self.oc_weight = nn.Parameter(torch.zeros(num_hidden, height, width))

        # layer_norm: 是否使用层归一化
        # 是卷积层，用于将隐藏状态降维，接在第 0 层之后，用于后续的 motion_list 和 dec_list。
        if layer_norm:
            self.conv_t_cc = nn.Sequential(
                nn.Conv2d(
                    in_channel,
                    num_hidden * 3,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=self.padding,
                    bias=False), nn.LayerNorm([num_hidden * 3, height, width]))
            self.conv_s_cc = nn.Sequential(
                nn.Conv2d(
                    num_hidden,
                    num_hidden * 4,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=self.padding,
                    bias=False), nn.LayerNorm([num_hidden * 4, height, width]))
            self.conv_x_cc = nn.Sequential(
                nn.Conv2d(
                    in_channel,
                    num_hidden * 4,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=self.padding,
                    bias=False), nn.LayerNorm([num_hidden * 4, height, width]))
            self.conv_h_concat = nn.Sequential(
                nn.Conv2d(
                    num_hidden,
                    num_hidden * 4,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=self.padding,
                    bias=False), nn.LayerNorm([num_hidden * 4, height, width]))
            self.conv_x_concat = nn.Sequential(
                nn.Conv2d(
                    num_hidden,
                    num_hidden * 4,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=self.padding,
                    bias=False), nn.LayerNorm([num_hidden * 4, height, width]))
        else:
            # 不开启层归一化
            # conv_t_cc: 卷积层，用于将隐藏状态降维，接在第 0 层之后，用于后续的 motion_list 和 dec_list。
            # conv_s_cc: 卷积层，用于将细胞状态降维，接在第 0 层之后，用于后续的 motion_list 和 dec_list。
            # conv_x_cc: 卷积层，用于将输入帧降维，接在第 0 层之后，用于后续的 motion_list 和 dec_list。
            # conv_h_concat: 卷积层，用于将隐藏状态和细胞状态拼接后，再进行卷积，得到门控信号。
            # conv_x_concat: 卷积层，用于将输入帧和隐藏状态拼接后，再进行卷积，得到门控信号。
            # conv_last: 卷积层，用于将隐藏状态和细胞状态拼接后，再进行卷积，得到门控信号。
            # attention_layer: Attention 层，用于将隐藏状态和细胞状态进行注意力机制处理。
            # global_feature_extractor: S-GFM 层，用于将输入帧提取全局特征。
            # gate: 门控层，用于将隐藏状态和全局特征进行门控处理。
            # conv_g: 卷积层，用于将隐藏状态升维，接在 motion_list 之后，用于后续的 gate_list。
            # MIMS: MIMS 层，用于将隐藏状态的差分转换为细胞状态的差分。0~N-1层需要差分，N层不需要差分。
            # forward: 前向传播，用于将输入帧、隐藏状态、细胞状态、运动高速公路进行处理，得到新的隐藏状态、细胞状态、运动高速公路。
            self.conv_t_cc = nn.Sequential(
                nn.Conv2d(
                    in_channel,
                    num_hidden * 3,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=self.padding,
                    bias=False), )
            self.conv_s_cc = nn.Sequential(
                nn.Conv2d(
                    num_hidden,
                    num_hidden * 4,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=self.padding,
                    bias=False), )
            self.conv_x_cc = nn.Sequential(
                nn.Conv2d(
                    in_channel,
                    num_hidden * 4,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=self.padding,
                    bias=False), )
            self.conv_h_concat = nn.Sequential(
                nn.Conv2d(
                    num_hidden,
                    num_hidden * 4,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=self.padding,
                    bias=False), )
            self.conv_x_concat = nn.Sequential(
                nn.Conv2d(
                    num_hidden,
                    num_hidden * 4,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=self.padding,
                    bias=False), )

        self.conv_last = nn.Conv2d(
            num_hidden * 2,
            num_hidden,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)

        # 是 Attention 层，用于将隐藏状态和细胞状态进行注意力机制处理。
        self.attention_layer = SelfAttentionMemory(
            num_hidden, att_hidden)
        # 是 S-GFM 层，用于将输入帧提取全局特征。
        self.global_feature_extractor = SpectralGlobalFocus(
            in_channel, num_hidden, height, width)
        # 是门控层，用于将隐藏状态和全局特征进行门控处理。
        self.gate = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=3, padding=1),
            nn.Sigmoid())
        if layer_norm:
            # 如果开启层归一化，则使用层归一化。
            self.conv_g = nn.Sequential(
                MultiScaleConv(
                    in_channel, num_hidden, kernel_sizes=[1, 3, 5], stride=1),
                nn.LayerNorm([num_hidden, height, width]))
        else:
            # 如果未开启层归一化，则不使用层归一化。
            self.conv_g = nn.Sequential(
                MultiScaleConv(
                    in_channel, num_hidden, kernel_sizes=[1, 3, 5],
                    stride=1), )

    def _init_state(self, inputs):
        return torch.zeros_like(inputs)

    def MIMS(self, x, h_t, c_t):
        if h_t is None:
            h_t = self._init_state(x)
        if c_t is None:
            c_t = self._init_state(x)
        Bh, Bc = h_t.shape[0], c_t.shape[0]
        if Bh != Bc:
            if Bh < Bc:
                pad = torch.zeros(
                    Bc - Bh,
                    *h_t.shape[1:],
                    dtype=h_t.dtype,
                    device=h_t.device)
                h_t = torch.cat([h_t, pad], dim=0)
            else:
                pad = torch.zeros(
                    Bh - Bc,
                    *c_t.shape[1:],
                    dtype=c_t.dtype,
                    device=c_t.device)
                c_t = torch.cat([c_t, pad], dim=0)

        h_concat = self.conv_h_concat(h_t)
        i_h, g_h, f_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        ct_act = torch.mul(c_t.repeat(1, 2, 1, 1), self.ct_weight)
        i_c, f_c = torch.split(ct_act, self.num_hidden, dim=1)
        i_ = i_h + i_c
        f_ = f_h + f_c
        g_ = g_h
        o_ = o_h
        if x is not None:
            x_concat = self.conv_x_concat(x)
            i_x, g_x, f_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
            i_ = i_ + i_x
            f_ = f_ + f_x
            g_ = g_ + g_x
            o_ = o_ + o_x
        i_ = torch.sigmoid(i_)
        f_ = torch.sigmoid(f_ + self._forget_bias)
        c_new = f_ * c_t + i_ * torch.tanh(g_)
        o_c = torch.mul(c_new, self.oc_weight)
        h_new = torch.sigmoid(o_ + o_c) * torch.tanh(c_new)
        return h_new, c_new

    def forward(self, x_t, diff_h, h_t, c_t, m_t, motion_highway):
        # 初始化隐藏状态、细胞状态、m_t是记忆状态mem，diff_h是隐藏状态的差分。
        h_t = self._init_state(x_t) if h_t is None else h_t
        c_t = self._init_state(x_t) if c_t is None else c_t
        m_t = self._init_state(x_t) if m_t is None else m_t
        diff_h = self._init_state(x_t) if diff_h is None else diff_h

        if getattr(
                self, 'convlstm_c',
                None) is not None and self.convlstm_c.shape[0] != x_t.shape[0]:
            self.convlstm_c = None

        # 卷积第一层：将隐藏状态、细胞状态、输入帧进行卷积，得到门控信号。
        t_cc = self.conv_t_cc(h_t)
        s_cc = self.conv_s_cc(m_t)
        x_cc = self.conv_x_cc(x_t)
        # 拆分门控信号。
        i_s, g_s, f_s, o_s = torch.split(s_cc, self.num_hidden, dim=1)
        i_t, g_t, o_t = torch.split(t_cc, self.num_hidden, dim=1)
        i_x, g_x, f_x, o_x = torch.split(x_cc, self.num_hidden, dim=1)

        # 计算门控信号。
        i = torch.sigmoid(i_x + i_t)
        i_ = torch.sigmoid(i_x + i_s)
        g = torch.tanh(g_x + g_t)
        g_ = torch.tanh(g_x + g_s)
        f_ = torch.sigmoid(f_x + f_s + self._forget_bias)
        o = torch.sigmoid(o_x + o_t + o_s)
        new_m = f_ * m_t + i_ * g_

        # 计算细胞状态的差分。
        c, self.convlstm_c = self.MIMS(
            diff_h, c_t, self.convlstm_c
            if self.convlstm_c is None else self.convlstm_c.detach())
        new_c = c + i * g
        cell = torch.cat((new_c, new_m), 1)
        new_h = o * torch.tanh(self.conv_last(cell))

        g_concat = self.conv_g(x_t)
        global_features = self.global_feature_extractor(x_t)
        gate = self.gate(torch.cat([new_h, global_features], dim=1))
        new_h = new_h * gate + global_features * (1 - gate)
        new_h = new_h * g_concat
        new_h, new_m = self.attention_layer(new_h, new_m)
        new_h = new_h + (1 - o) * motion_highway
        motion_highway = new_h
        return new_h, new_c, new_m, motion_highway
