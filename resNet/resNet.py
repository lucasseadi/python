import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode="fan out")
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        # shortcut / flashback
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module("conv",
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               bias=False))

    def forward(self, x):
        y = self.bn1(x)                         # standardizes the input
        y = F.relu(y, inplace=True)             # zeroes negative values
        y = self.conv1(y)                       # convolution: in_channels to out_channels
        y = F.relu(self.bn2(y), inplace=True)   # zeroes negative values
        y = self.conv2(y)                       # convolution: out_channels to out_channels
        y += self.shortcut(x)                   # convolution: in_channels to out_channels
        return y


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride):
        super(BottleneckBlock, self).__init__()
        # reduces the data size every time
        bottleneckBlock_channels = out_channels // self.expansion

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bottleneckBlock_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.bn2 = nn.BatchNorm2d(bottleneckBlock_channels)
        self.conv2 = nn.Conv2d(bottleneckBlock_channels, bottleneckBlock_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)

        self.bn3 = nn.BatchNorm2d(bottleneckBlock_channels)
        self.conv3 = nn.Conv2d(bottleneckBlock_channels, out_channels, kernel_size=1, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module("conv", nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                                                       padding=0, bias=False))

    def forward(self, x):
        y = self.bn1(x)                         # standardizes the input
        y = F.relu(y, inplace=True)             # zeroes negative values
        y = self.conv1(y)                       # convolution: in_channels to bottleneckBlock_channels
        y = F.relu(self.bn2(y), inplace=True)   # zeroes negative values
        y = self.conv2(y)                       # convolution: bottleneckBlock_channels to bottleneckBlock_channels
        y = F.relu(self.bn3(y), inplace=True)   # zeroes negative values
        y = self.conv3(y)                       # convolution: bottleneckBlock_channels to out_channels
        y += self.shortcut(x)                   # convolution: in_channels to out_channels
        return y


class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        input_shape = config["input_shape"]
        base_channels = config["base_channels"]
        block_type = config["block_type"]
        depth = config["depth"]

        assert block_type in ["basic", "bottleneck"]
        if block_type == "basic":
            block = BasicBlock
            # number of blocks per layer
            n_blocks_per_stage = (depth - 2) // 6
            assert (n_blocks_per_stage * 6 + 2) == depth
        else:
            block = BottleneckBlock
            n_blocks_per_stage = (depth - 2) // 9
            assert (n_blocks_per_stage * 9 + 2) == depth

        # n_channels = [16, 128, 256]
        n_channels = [base_channels, base_channels * 2 * block.expansion, base_channels * 4 * block.expansion]

        self.conv = nn.Conv2d(input_shape[1], n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(n_channels[2])

        self.stage1 = self._make_stage(n_channels[0], n_channels[0], n_blocks_per_stage, block, stride=1)
        self.stage2 = self._make_stage(n_channels[0], n_channels[1], n_blocks_per_stage, block, stride=2)
        self.stage3 = self._make_stage(n_channels[1], n_channels[2], n_blocks_per_stage, block, stride=2)

    def _make_stage(self, in_channels, out_channels, n_blocks, block, stride):
        stage = nn.Sequential()
        for index in range(n_blocks):
            block_name = "block{}".format(index + 1)
            if index == 0:
                stage.add_module(block_name, block(in_channels, out_channels, stride=stride))
            else:
                stage.add_module(block_name, block(out_channels, out_channels, stride=1))
        return stage

    def _forward_conv(self, x):
        x = self.conv(x)                                # convolution: 3 to 16
        x = self.stage1(x)                              # 3 BasicBlocks / 2 BottleneckBlocks 16 to 16
        x = self.stage2(x)                              # 3 BasicBlocks / 2 BottleneckBlocks 16 to 128
        x = self.stage3(x)                              # 3 BasicBlocks / 2 BottleneckBlocks 128 to 256
        # apply BN and ReLU before average pooling
        x = F.relu(self.bn(x), inplace=True)            # standardizes the input, zeroes negative values
        x = F.adaptive_avg_pool2d(x, output_size=1)     # calculates tha average value for each patch
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        return x
