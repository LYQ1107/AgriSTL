import torch
import torch.nn as nn
import torch.nn.functional as F


_backwarp_grids = {}


def warp(ten_input: torch.Tensor, ten_flow: torch.Tensor) -> torch.Tensor:
    key = (str(ten_flow.device), tuple(ten_flow.shape))
    if key not in _backwarp_grids:
        ten_horizontal = torch.linspace(
            -1.0, 1.0, ten_flow.shape[3], device=ten_flow.device, dtype=ten_flow.dtype
        ).view(1, 1, 1, ten_flow.shape[3]).expand(ten_flow.shape[0], -1, ten_flow.shape[2], -1)
        ten_vertical = torch.linspace(
            -1.0, 1.0, ten_flow.shape[2], device=ten_flow.device, dtype=ten_flow.dtype
        ).view(1, 1, ten_flow.shape[2], 1).expand(ten_flow.shape[0], -1, -1, ten_flow.shape[3])
        _backwarp_grids[key] = torch.cat([ten_horizontal, ten_vertical], 1)

    ten_flow = torch.cat(
        [
            ten_flow[:, 0:1] / ((ten_input.shape[3] - 1.0) / 2.0),
            ten_flow[:, 1:2] / ((ten_input.shape[2] - 1.0) / 2.0),
        ],
        1,
    )
    grid = (_backwarp_grids[key] + ten_flow).permute(0, 2, 3, 1)
    return F.grid_sample(
        ten_input, grid, mode='bilinear', padding_mode='border', align_corners=True
    )


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.bernoulli(x)

    @staticmethod
    def backward(ctx, grad):
        return grad


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.PReLU(out_planes),
    )


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        nn.PReLU(out_planes),
    )


class MVFB(nn.Module):
    def __init__(self, in_planes, num_feature):
        super().__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, num_feature // 2, 3, 2, 1),
            conv(num_feature // 2, num_feature, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            conv(num_feature, num_feature),
            conv(num_feature, num_feature),
            conv(num_feature, num_feature),
        )
        self.conv_sq = conv(num_feature, num_feature // 4)

        self.conv1 = nn.Sequential(conv(in_planes, 8, 3, 2, 1))
        self.convblock1 = nn.Sequential(conv(8, 8))
        self.lastconv = nn.ConvTranspose2d(num_feature // 4 + 8, 5, 4, 2, 1)

    def forward(self, x, flow, scale):
        x0 = x
        flow0 = flow
        if scale != 1:
            x = F.interpolate(x, scale_factor=1.0 / scale, mode='bilinear', align_corners=False)
            flow = (
                F.interpolate(flow, scale_factor=1.0 / scale, mode='bilinear', align_corners=False)
                * (1.0 / scale)
            )
        x = torch.cat((x, flow), 1)
        x1 = self.conv0(x)
        x2 = self.conv_sq(self.convblock(x1) + x1)
        x2 = F.interpolate(x2, scale_factor=scale * 2, mode='bilinear', align_corners=False)

        x3 = self.conv1(torch.cat((x0, flow0), 1))
        x4 = self.convblock1(x3)
        tmp = self.lastconv(torch.cat((x2, x4), dim=1))
        flow = tmp[:, :4]
        mask = tmp[:, 4:5]
        return flow, mask


class DMVFNCore(nn.Module):
    def __init__(self, num_features=(160, 160, 160, 80, 80, 80, 44, 44, 44)):
        super().__init__()
        self.blocks = nn.ModuleList([MVFB(13 + 4, num_feature=nf) for nf in num_features])
        self.routing = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.l1 = nn.Linear(32, len(num_features))

    def forward(self, pair: torch.Tensor, scale_list):
        batch_size, _, height, width = pair.shape
        routing_vector = self.routing(pair[:, :6]).reshape(batch_size, -1)
        routing_vector = torch.sigmoid(self.l1(routing_vector))
        routing_vector = routing_vector / (routing_vector.sum(1, keepdim=True) + 1e-6) * 4.5
        routing_vector = torch.clamp(routing_vector, 0, 1)
        ref = RoundSTE.apply(routing_vector)

        img0 = pair[:, :3]
        img1 = pair[:, 3:6]
        warped_img0 = img0
        warped_img1 = img1
        flow = pair.new_zeros(batch_size, 4, height, width)
        mask = pair.new_zeros(batch_size, 1, height, width)

        merged_final = []
        mask_final = []
        for i, block in enumerate(self.blocks):
            block_input = torch.cat((img0, img1, warped_img0, warped_img1, mask), 1)
            flow_d, mask_d = block(block_input, flow, scale=scale_list[i])

            flow_right_now = flow + flow_d
            mask_right_now = mask + mask_d

            routing_weight = ref[:, i].reshape(batch_size, 1, 1, 1)
            flow = flow + flow_d * routing_weight
            mask = mask + mask_d * routing_weight

            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            warped_img0_right_now = warp(img0, flow_right_now[:, :2])
            warped_img1_right_now = warp(img1, flow_right_now[:, 2:4])

            if i < len(self.blocks) - 1:
                mask_final.append(torch.sigmoid(mask_right_now))
                merged_final.append((warped_img0_right_now, warped_img1_right_now))
            else:
                mask_final.append(torch.sigmoid(mask))
                merged_final.append((warped_img0, warped_img1))

        outputs = []
        for i in range(len(merged_final)):
            pred = merged_final[i][0] * mask_final[i] + merged_final[i][1] * (1 - mask_final[i])
            outputs.append(torch.clamp(pred, 0, 1))
        return outputs


class DMVFN_Model(nn.Module):
    r"""DMVFN next-frame predictor wrapped for AgriSTL autoregressive rollout."""

    def __init__(
        self,
        in_shape,
        scale_list=(4, 4, 4, 2, 2, 2, 1, 1, 1),
        num_features=(160, 160, 160, 80, 80, 80, 44, 44, 44),
        **kwargs,
    ):
        super().__init__()
        _, c, _, _ = in_shape
        assert c == 3, 'DMVFN currently expects RGB inputs with 3 channels.'
        assert len(scale_list) == len(num_features), 'scale_list and num_features must have the same length.'
        self.scale_list = list(scale_list)
        self.core = DMVFNCore(num_features=num_features)

    def forward(self, x_raw: torch.Tensor, return_all_stages: bool = False, **kwargs) -> torch.Tensor:
        assert x_raw.ndim == 5, 'DMVFN_Model expects input of shape [B, T, C, H, W].'
        assert x_raw.shape[1] >= 2, 'DMVFN requires at least two context frames.'
        pair = torch.cat([x_raw[:, -2], x_raw[:, -1]], dim=1)
        stage_preds = self.core(pair, self.scale_list)
        if return_all_stages:
            return stage_preds
        return stage_preds[-1].unsqueeze(1)
