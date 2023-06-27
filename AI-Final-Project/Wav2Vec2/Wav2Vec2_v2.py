# The large model pretrained and fine-tuned on 960 hours of Librispeech on 16kHz sampled speech audio.
# When using the model make sure that your speech input is also sampled at 16Khz.

import os
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer

# Define paths to your fine-tuning data
train_audio_path = "path/to/train/audio"
train_transcripts_path = "path/to/train/transcripts.txt"
validation_audio_path = "path/to/validation/audio"
validation_transcripts_path = "path/to/validation/transcripts.txt"

# Load the pretrained model and tokenizer
model_name = "facebook/wav2vec2-large-960h"
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Set the model in training mode
model.train()


# Define your dataset class to load and process the audio and transcripts
class AudioTranscriptDataset(torch.utils.data.Dataset):
    def __init__(self, audio_dir, transcripts_file, tokenizer):
        self.audio_dir = audio_dir
        self.transcripts = []
        self.tokenizer = tokenizer

        with open(transcripts_file, "r") as f:
            for line in f:
                transcript = line.strip()
                self.transcripts.append(transcript)

    def __len__(self):
        return len(self.transcripts)

    def __getitem__(self, index):
        transcript = self.transcripts[index]
        audio_file = os.path.join(self.audio_dir, f"{index}.wav")
        waveform, _ = torchaudio.load(audio_file)

        # Preprocess the audio and transcript
        inputs = self.tokenizer(transcript, padding="max_length", truncation=True, return_tensors="pt")
        input_values = inputs.input_values.squeeze()
        labels = inputs.input_ids.squeeze()

        return input_values, labels


# Create dataset and data loaders
train_dataset = AudioTranscriptDataset(train_audio_path, train_transcripts_path, tokenizer)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

validation_dataset = AudioTranscriptDataset(validation_audio_path, validation_transcripts_path, tokenizer)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=4)

# Define the optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop

for epoch in range(10):  # Number of epochs to train
    for batch in train_loader:
        input_values, labels = batch

        # Forward pass
        outputs = model(input_values=input_values, labels=labels)

        # Compute loss
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    with torch.no_grad():
        for batch in validation_loader:
            input_values, labels = batch
            outputs = model(input_values=input_values, labels=labels)
            # Compute metrics or perform other validation operations

    # Adjust learning rate
    scheduler.step()

# Save the fine-tuned model
output_dir = "path/to/save/fine-tuned/model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
