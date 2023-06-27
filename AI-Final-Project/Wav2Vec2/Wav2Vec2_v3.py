# The large model pretrained and fine-tuned on 960 hours of Librispeech on 16kHz sampled speech audio.
# When using the model make sure that your speech input is also sampled at 16Khz.

# Using MyLibriSpeech as dataset class

import random
from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer

from MyLibriSpeech import *
from utils import *

# Load the pretrained model and tokenizer
model_name = "facebook/wav2vec2-large-960h"
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Set the model in training mode
model.train()


def load_data(config):
    print(f"Loading dataset...")
    train_set, val_set = Dataset(), Dataset()

    max_audio_length = 392400
    max_transcript_length = 398
    dataset_path = os.path.dirname(os.path.abspath(__file__)) + "/../data/librispeech/"
    char2index = {'D': 0, 'G': 1, 'O': 2, 'S': 3, 'J': 4, 'T': 5, ' ': 6, 'E': 7, 'X': 8, 'A': 9, 'K': 10, 'R': 11,
                  'N': 12, 'L': 13, 'U': 14, 'C': 15, 'I': 16, 'M': 17, "'": 18, 'H': 19, 'Q': 20, 'B': 21, 'Y': 22,
                  'Z': 23, 'W': 24, 'F': 25, 'P': 26, 'V': 27}

    # randomly defines which entries will go to train_set and val_set
    # librispeech train-clean-100 has 28539 entries
    # 28539 * 0.02 = 570
    train_val_lines = list(range(1, 28540))

    global vocabulary
    vocabulary = tokenize_transcripts_librispeech(config, dataset_path, train_val_lines, char2index)

    random.shuffle(train_val_lines)
    train_lines = train_val_lines[570:]
    val_lines = train_val_lines[:570]

    train_set = MyLibriSpeech(dataset_path, url="train-clean-100", download=True, dataset_lines=train_lines,
                              char2index=char2index)
    val_set = MyLibriSpeech(dataset_path, url="train-clean-100", download=True, dataset_lines=val_lines,
                            char2index=char2index)

    print("train_set", len(train_set))
    print("val_set", len(val_set))

    train_set.max_audio_length = max_audio_length
    train_set.max_transcript_length = max_transcript_length

    val_set.max_audio_length = max_audio_length
    val_set.max_transcript_length = max_transcript_length

    return train_set, val_set, max_audio_length


config = {"batch_size": 1, "epochs": 10, "learning_rate": 0.001, "l1": 128, "l2": 64, "dataset": "librispeech"}

# Create dataset and data loaders
train_set, val_set, max_audio_length = load_data(config)
train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=True)

# Define the optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop

print("Will start training...")

for epoch in range(10):  # Number of epochs to train
    for batch in train_loader:
        input_values, labels = batch
        print("input_values (1)", input_values.shape)

        # Reshape the input tensor
        input_values = input_values.squeeze(1)  # Remove the second dimension
        print("input_values (2)", input_values.shape)

        # Forward pass
        outputs = model(input_values=input_values, labels=labels)

        # Compute loss
        loss = outputs.loss

        print(f"Epoch[{epoch}/10]: loss: {loss}")

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    with torch.no_grad():
        for batch in val_loader:
            input_values, labels = batch
            outputs = model(input_values=input_values, labels=labels)
            # Compute metrics or perform other validation operations

    # Adjust learning rate
    scheduler.step()

# Save the fine-tuned model
output_dir = "model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
