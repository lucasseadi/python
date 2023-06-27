import logging
import torch
import torch.nn as nn


class TranscriptionNet(nn.Module):
    def __init__(self, config, vocabulary, max_audio_length):
        super(TranscriptionNet, self).__init__()

        self.logger = logging.getLogger("FinalProject")
        self.num_classes = len(vocabulary)
        self.vocabulary = vocabulary

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.logger.info(f"Initializing self.rnn...")

        # Recurrent layers
        # self.rnn = nn.GRU(64 * 50, 128, batch_first=True)
        # self.rnn = nn.GRU(490, 128, batch_first=True)
        # self.rnn = nn.GRU(64, 128, batch_first=True)
        # self.rnn = nn.GRU(64 * max_audio_length // 4, 128, batch_first=True)
        # self.rnn = nn.GRU(64 * (max_audio_length // 4) * (max_audio_length // 4), 128, batch_first=True)
        # self.rnn = nn.GRU(64 * (max_audio_length // 4) * (max_audio_length // 4), 32, batch_first=True)
        # self.rnn = nn.GRU(64 * (max_audio_length // 1024) * (max_audio_length // 1024), 32, batch_first=True)
        self.rnn = nn.GRU(64 * (max_audio_length // 1024) * (max_audio_length // 1024), 32, batch_first=True)

        self.logger.info(f"Initializing self.fc...")

        # Fully connected layer
        # self.fc = nn.Linear(128, self.num_classes)
        self.fc = nn.Linear(32, self.num_classes)

    def forward(self, x):
        self.logger.info(f"[forward] x (input) {x.shape}")

        # Apply convolutional layers
        x = self.conv_layers(x)
        self.logger.info(f"[forward] x (conv_layers) {x.shape}")

        # Reshape the tensor for the recurrent layers
        batch_size, channels, freq, time = x.size()
        self.logger.info(f"[forward] batch_size {batch_size}")
        self.logger.info(f"[forward] channels {channels}")
        self.logger.info(f"[forward] freq {freq}")
        self.logger.info(f"[forward] time {time}")
        # x = x.view(batch_size, channels * freq, time)
        # x = x.view(batch_size, time, channels * freq)
        # x = x.view(batch_size, time, -1)
        # x = x.view(batch_size * time, -1)
        # x = x.view(batch_size * time, channels * freq)

        # x = x.permute(0, 3, 1, 2)  # Swap dimensions for time and channels
        # x = x.permute(0, 2, 1, 3)
        x = x.permute(0, 2, 3, 1)
        self.logger.info(f"[forward] x (permute) {x.shape}")

        # x = x.reshape(batch_size * time, channels * freq)
        # x = x.reshape(batch_size, freq, -1)
        # x = x.reshape(batch_size * time, channels, freq)
        x = x.reshape(batch_size, time, -1)
        self.logger.info(f"[forward] x (reshape) {x.shape}")

        # Apply recurrent layers
        # _, hidden = self.rnn(x)
        # self.logger.info(f"[forward] hidden (rnn) {hidden.shape}")
        # hidden = hidden.squeeze(0)
        # self.logger.info(f"[forward] hidden (squeeze) {hidden.shape}")
        x, _ = self.rnn(x)
        self.logger.info(f"[forward] x (rnn) {x.shape}")

        # Reshape the output tensor
        x = x.reshape(batch_size, -1)
        # x = x.reshape(batch_size, time, -1)
        self.logger.info(f"[forward] x (reshape) {x.shape}")

        # Apply fully connected layer
        # x = self.fc(hidden)
        x = self.fc(x)
        self.logger.info(f"[forward] x (fc) {x.shape}")

        # Apply softmax activation
        # x = torch.softmax(x, dim=1)

        # Reshape back to batched sequence
        # x = x.view(batch_size, time, -1)
        # x = x.reshape(batch_size, time, -1)
        # x = x.reshape(batch_size, -1, self.num_classes)
        # self.logger.info(f"[forward] x (reshape) {x.shape}")

        # Get the predicted words for each time step
        # _, predicted_indices = torch.max(x, dim=2)
        # predicted_words = [self.vocabulary[idx.item()] for idx in predicted_indices]

        # Get the predicted words for each input in the batch
        predicted_words = []
        for i in range(batch_size):
            _, predicted_indices = torch.max(x[i], dim=1)
            predicted_words.append([self.vocabulary[idx.item()] for idx in predicted_indices])

        self.logger.info(f"[forward] predicted_words {predicted_words}")

        return predicted_words
