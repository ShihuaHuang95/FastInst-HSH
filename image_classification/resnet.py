from layers import ResNetBottleneckBlock, IdentityLayer, ResidualBlock, ConvLayer, LinearLayer
from utils import MyGlobalAvgPool2d, make_divisible

import torch.nn as nn
BASE_DEPTH_LIST = [2, 2, 4, 2]
STAGE_WIDTH_LIST = [256, 512, 1024, 2048]
CHANNEL_DIVISIBLE = 8


class ResNet50(nn.Module):
    def __init__(
        self,
        n_classes=1000,
        bn_param=(0.1, 1e-5),
        dropout_rate=0,
        depth_list=[2, 0, 1, 1, 1],
        expand_ratio_list=[0.25, 0.2, 0.2, 0.2, 0.2, 0.25, 0.25, 0.35, 0.2, 0.2, 0.25, 0.2, 0.2, 0.2, 0.2, 0.25, 0.2, 0.25],
        width_mult_list=[0, 1, 0, 0, 1, 2],
    ):
        super(ResNet50, self).__init__()   
        BaseWidthList = [0.65, 0.8, 1.0]
        width_mult = 1
        input_channel = make_divisible(64 * BaseWidthList[width_mult_list[0]], CHANNEL_DIVISIBLE)
        mid_input_channel = make_divisible(input_channel // 2, CHANNEL_DIVISIBLE)
        input_channel = make_divisible(64 * BaseWidthList[width_mult_list[1]], CHANNEL_DIVISIBLE)

        # build input stem
        if depth_list[0] == 2:
            input_stem = [
                ConvLayer(3, mid_input_channel, 3, stride=2, use_bn=True, act_func="relu"),
                ResidualBlock(
                    ConvLayer(
                        mid_input_channel,
                        mid_input_channel,
                        3,
                        stride=1,
                        use_bn=True,
                        act_func="relu",
                    ),
                    IdentityLayer(mid_input_channel, mid_input_channel),
                ),
                ConvLayer(
                    mid_input_channel,
                    input_channel,
                    3,
                    stride=1,
                    use_bn=True,
                    act_func="relu",
                ),
            ]
        else:
            input_stem = [
                ConvLayer(3, mid_input_channel, 3, stride=2, use_bn=True, act_func="relu"),
                ConvLayer(
                    mid_input_channel,
                    input_channel,
                    3,
                    stride=1,
                    use_bn=True,
                    act_func="relu",
                ),
            ]      

        stage_width_list = STAGE_WIDTH_LIST.copy()
        for i, width in enumerate(stage_width_list):
            stage_width_list[i] = make_divisible(width * BaseWidthList[width_mult_list[i+2]], CHANNEL_DIVISIBLE)

        new_depth_list = []
        new_expand_ratio_list = []
        start = 0
        for base, depth in zip(BASE_DEPTH_LIST, depth_list[1:]):
            stage_len = base + depth
            end = start + base + 2
            new_depth_list.append(stage_len)
            stage_expand_ratio_list = expand_ratio_list[start:end]       # 
            stage_e = []
            for i in range(base + depth):
                stage_e.append(stage_expand_ratio_list[i])
            new_expand_ratio_list.append(stage_e)
            start = end

        stride_list = [1, 2, 2, 2]
        expand_ratio = 0.25
        self.stage_idx = [sum(new_depth_list[:i+1]) for i in range(len(new_depth_list))]
        print(new_depth_list, new_expand_ratio_list, self.stage_idx)
        # blocks
        blocks = []
        stage = 0
        for d, width, s in zip(new_depth_list, stage_width_list, stride_list):
            for i in range(d):
                stride = s if i == 0 else 1
                bottleneck_block = ResNetBottleneckBlock(
                    input_channel,
                    width,
                    kernel_size=3,
                    stride=stride,
                    expand_ratio=new_expand_ratio_list[stage][i],
                    act_func="relu",
                    downsample_mode="avgpool_conv",
                )
                blocks.append(bottleneck_block)
                input_channel = width
            stage += 1

        self.input_stem = nn.ModuleList(input_stem)
        self.max_pooling = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )
        self.blocks = nn.ModuleList(blocks)
        self.global_avg_pool = MyGlobalAvgPool2d(keep_dim=False)
        # classifier
        self.classifier = LinearLayer(input_channel, n_classes)

    def forward(self, x):
        for layer in self.input_stem:
            x = layer(x)
        x = self.max_pooling(x)
        outputs = {}
        idx = 0
        for i, block in enumerate(self.blocks):
            x = block(x)
            if (i+1) in self.stage_idx:
                key = 'res{}'.format(idx + 2)
                outputs[key] = x
                idx += 1
        x = self.global_avg_pool(x)
        x = self.classifier(x)
        print(outputs['res2'].shape, outputs['res4'].shape, outputs['res5'].shape)
        return x

if __name__ == "__main__":
    import torch
    pt_path="./ResNet50D-Params@17.40-FLOPs@1924M.pth"
    net = ResNet50()
    # print(net)
    state_dicts = torch.load(pt_path, map_location="cpu")
    net.load_state_dict(state_dicts, strict=True)
    print("     ## Pretrained Weights Loading Successful!!! ###")
