import logging
import torch.nn as nn
from torch import Tensor

import torch
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, config, num_classes):
        super(CNN, self).__init__()

        self.logger = logging.getLogger("FinalProject")
        self.batch_size = config["batch_size"]
        self.num_classes = num_classes
        l1 = config["l1"]
        l2 = config["l2"]

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.batch_size, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=self.batch_size, out_channels=2 * self.batch_size, kernel_size=3, stride=1,
                               padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.linear1 = nn.Linear((2 * self.batch_size) * 50 * 490, l1)
        self.linear2 = nn.Linear(l1, l2)
        # self.linear3 = nn.Linear(l2, 2)
        # self.linear3 = nn.Linear(l2, 398)   # max_transcript_length
        self.linear3 = nn.Linear(l2, self.num_classes)   # max_transcript_length

    def preview_shape(self, x, layer):
        # print("[preview_shape] x", x.shape)

        # x.shape [batch_size, in_channels, height, width]
        # layer.weight.shape (out_channels, in_channels, kernel_height, kernel_width)
        input_height = x.shape[2]
        input_width = x.shape[3]

        # output.shape [in_channels, out_channels, (input_height - kernel_height + 2*padding) / stride + 1,
        # (input_width - kernel_width + 2*padding) / stride + 1]
        if isinstance(layer, nn.Conv2d):
            layer_type = "nn.Conv2d"
            in_channels = layer.weight.shape[0]
            # out_channels = layer.weight.shape[1]
            out_channels = x.shape[0]
            kernel_height = layer.weight.shape[2]
            kernel_width = layer.weight.shape[3]
            padding = layer.padding[0]
            stride = layer.stride[0]
            output_height = int((input_height - kernel_height + 2 * padding) / stride + 1)
            output_width = int((input_width - kernel_width + 2 * padding) / stride + 1)

        # output.shape [in_channels, out_channels, (input_height - kernel_height) / stride + 1,
        # (input_width - kernel_width) / stride + 1]
        elif isinstance(layer, nn.MaxPool2d):
            layer_type = "nn.MaxPool2d"
            in_channels = x.shape[1]
            out_channels = x.shape[0]
            kernel_height = layer.kernel_size
            kernel_width = layer.kernel_size
            stride = layer.stride
            output_height = int((input_height - kernel_height) / stride + 1)
            output_width = int((input_width - kernel_width) / stride + 1)

        elif isinstance(layer, nn.Linear):
            pass

        shape = f"[{out_channels}, {in_channels}, {output_height}, {output_width}]"
        self.logger.info(f"NEW SHAPE AFTER {layer_type} SHOULD BE {shape}")

    def forward(self, x: Tensor) -> Tensor:
        # Convolutional layers
        self.logger.info(f"[forward] x (input) {x.shape}")

        self.preview_shape(x, self.conv1)
        x = self.relu(self.conv1(x))
        self.logger.info(f"[forward] x (conv1) {x.shape}")

        self.preview_shape(x, self.pool)
        x = self.pool(x)
        self.logger.info(f"[forward] x (pool) {x.shape}")

        self.preview_shape(x, self.conv2)
        x = self.relu(self.conv2(x))
        self.logger.info(f"[forward] x (conv2) {x.shape}")

        self.preview_shape(x, self.pool)
        x = self.pool(x)
        self.logger.info(f"[forward] x (pool) {x.shape}")

        # x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = x.view(x.size(0), -1)
        self.logger.info(f"[forward] x (view) {x.shape}")

        # Dense layers
        x = self.relu(self.linear1(x))
        self.logger.info(f"[forward] x (linear1) {x.shape}")

        x = self.relu(self.linear2(x))
        self.logger.info(f"[forward] x (linear2) {x.shape}")

        x = self.linear3(x)
        self.logger.info(f"[forward] x (linear3) {x.shape}")

        return x
