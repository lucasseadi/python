# loads the datasets and the pretrained nn_models with train test split
# training loop
# mse loss

import os
import random
import torch.nn as nn
from torch.utils.data import Dataset

from nn_models.CNN import CNN
from MyCommonVoice import MyCommonVoice
from utils import *


def load_data(dataset_name):
    logger.info(f"Loading {dataset_name} dataset...")
    train_set, val_set, test_set = Dataset(), Dataset(), Dataset()
    if dataset_name == "librispeech":
        max_audio_length = 392400
        max_transcript_length = 398
        dataset_path = os.path.dirname(os.path.abspath(__file__)) + "/data/librispeech/"

        # randomly defines which entries will go to train_set and val_set
        # librispeech train-clean-100 has 28539 entries
        # 28539 * 0.02 = 570
        train_val_lines = list(range(1, 28540))
        random.shuffle(train_val_lines)
        train_lines = train_val_lines[570:]
        val_lines = train_val_lines[:570]
        char2index = {'D': 0, 'G': 1, 'O': 2, 'S': 3, 'J': 4, 'T': 5, ' ': 6, 'E': 7, 'X': 8, 'A': 9, 'K': 10, 'R': 11,
                      'N': 12, 'L': 13, 'U': 14, 'C': 15, 'I': 16, 'M': 17, "'": 18, 'H': 19, 'Q': 20, 'B': 21, 'Y': 22,
                      'Z': 23, 'W': 24, 'F': 25, 'P': 26, 'V': 27}

        train_set = MyLibriSpeech(dataset_path, url="train-clean-100", download=True, dataset_lines=train_lines,
                                  char2index=char2index)
        val_set = MyLibriSpeech(dataset_path, url="train-clean-100", download=True, dataset_lines=val_lines,
                                char2index=char2index)
        test_set = MyLibriSpeech(dataset_path, url="test-clean", download=True, char2index=char2index)

    elif dataset_name == "commonvoice":
        dataset_path = 'data/cv-corpus-13.0-delta-2023-03-09/'
        max_audio_length = 352512
        max_transcript_length = 115

        train_set = MyCommonVoice(dataset_path, train=True)
        train_set.char2index = {'b': 0, 'o': 1, 'c': 2, '.': 3, 'd': 4, 'I': 5, 'X': 6, 'U': 7, '’': 8, 'z': 9, ':': 10,
                                ' ': 11, ',': 12, '?': 13, 'N': 14, 't': 15, 'x': 16, '—': 17, '‑': 18, 'Q': 19,
                                'm': 20, 'A': 21, 'J': 22, 'C': 23, 'P': 24, 'g': 25, '"': 26, ';': 27, 'r': 28,
                                'L': 29, 'h': 30, 'i': 31, 'y': 32, 'k': 33, 'K': 34, 'â': 35, 'e': 36, '”': 37,
                                'v': 38, 'M': 39, 'W': 40, '-': 41, 'T': 42, 's': 43, '(': 44, 'a': 45, 'Y': 46,
                                'l': 47, 'D': 48, '“': 49, 'é': 50, 'G': 51, 'B': 52, "'": 53, 'S': 54, 'q': 55,
                                'w': 56, 'E': 57, '–': 58, 'R': 59, '!': 60, 'j': 61, 'O': 62, 'p': 63, 'Z': 64,
                                'V': 65, 'n': 66, ')': 67, 'F': 68, 'f': 69, '‘': 70, 'u': 71, 'H': 72}

        val_set = MyCommonVoice(dataset_path, train=False)
        val_set.char2index = train_set.char2index
        val_set.val_set = train_set.val_set
        val_set.process_common_voice_val_set()
        val_set.char2index = train_set.char2index

        test_set = MyCommonVoice(dataset_path, train=False)
        test_set.test_set = train_set.test_set
        test_set.process_common_voice_test_set()
        test_set.char2index = train_set.char2index

    train_set.max_audio_length = max_audio_length
    train_set.max_transcript_length = max_transcript_length

    val_set.max_audio_length = max_audio_length
    val_set.max_transcript_length = max_transcript_length

    test_set.max_audio_length = max_audio_length
    test_set.max_transcript_length = max_transcript_length

    return train_set, val_set, test_set


def load_state_librispeech(config):
    checkpoint_path = os.path.dirname(os.path.abspath(__file__)) + "/data/librispeech_pretrained.pth"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model_state_dict = checkpoint["state_dict"]
    model = CNN(config)
    model.load_state_dict(model_state_dict, strict=False)
    return model


def load_state_commonvoice(config):
    logger.info("[load_state_commonvoice]")
    from speechbrain.pretrained import EncoderDecoderASR
    model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-wav2vec2-commonvoice-en",
                                           savedir="pretrained_models/asr-wav2vec2-commonvoice-en")
    return model


def load_model(config):
    logger.info(f"Loading {config['dataset']} state...")
    if config["dataset"] == "librispeech":
        model = load_state_librispeech(config)
    elif config["dataset"] == "commonvoice":
        model = load_state_commonvoice(config)
    return model


# data_dir arg is necessary for trial to work
def train_model(config, data_dir=None):
    ########## Loads data ##########
    train_set, val_set, test_set = load_data(config["dataset"])
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=True)

    ########## Loads pre-trained model ##########
    model = load_model(config)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    for epoch in range(config["epochs"]):  # loop over the dataset multiple times
        model.train()

        running_loss = 0.0

        # for i, data in enumerate(train_loader, 0):
        for batch in tqdm(train_loader, desc="Training model", unit="batch"):
            audios, transcripts = batch

            logger.info(f"[train_model] audio {audios.shape}")
            logger.info(f"[train_model] transcript {len(transcripts)}")

            # Add channel dimension for the CNN model
            audios = audios.unsqueeze(1)

            logger.info(f"[train_model] audio {audios.shape}")

            # Get the spectrogram for the audio
            spectrograms = get_spectrogram(audios)

            logger.info(f"[train_model] spectrograms {spectrograms.shape}")

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(spectrograms)

            # Adjust the transcripts batch size
            new_transcripts = []
            for transcript in transcripts:
                # logger.info("[train_model] transcript (1)", len(transcript))
                # transcript = transcript[:outputs.size(0)]  # Adjust batch size to match outputs
                # logger.info("[train_model] transcript (2)", len(transcript))
                # transcript = ''.join(transcript)
                # logger.info("[train_model] transcript (3)", len(transcript))

                # Convert the transcript to a tensor of Long type
                transcript = [train_set.char2index[c] for c in transcript]
                logger.info(f"[train_model] transcript (4) {len(transcript)}")
                transcript = torch.tensor(transcript, dtype=torch.long)
                logger.info(f"[train_model] transcript (5) {transcript.shape}")
                new_transcripts.append(transcript)
            # transcripts = torch.stack(new_transcripts)
            transcripts = torch.stack(new_transcripts).float()

            logger.info(f"[train_model] outputs {outputs.shape}")
            logger.info(f"[train_model] transcripts {transcripts.shape}")

            # input_lengths = torch.full((transcripts.size(0),), outputs.size(1), dtype=torch.long)
            # input_lengths = torch.full((outputs.size(0),), outputs.size(1), dtype=torch.long)
            # input_lengths = torch.full((transcripts.size(0),), outputs.size(0), dtype=torch.long)
            # input_lengths = torch.full((transcripts.size(0),), transcripts.size(1), dtype=torch.long)
            input_lengths = torch.full((transcripts.size(0),), outputs.size(1), dtype=torch.long).unsqueeze(1)

            # target_lengths = torch.tensor((outputs.size(0),), dtype=torch.long)
            # target_lengths = torch.tensor((outputs.size(1),), dtype=torch.long)
            # target_lengths = torch.full((transcripts.size(0),), outputs.size(1), dtype=torch.long)
            # target_lengths = torch.full((transcripts.size(0),), transcripts.size(1), dtype=torch.long)
            target_lengths = torch.tensor([len(t) for t in transcripts], dtype=torch.long)

            logger.info(f"[train_model] input_lengths {input_lengths.shape}")
            logger.info(f"[train_model] target_lengths {target_lengths.shape}")

            loss = criterion(outputs, transcripts)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_set)
        print(f"[Epoch {epoch + 1}/{config['epochs']}] Loss: {epoch_loss}")

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(val_loader, 0):
            with torch.no_grad():
                inputs, labels = data

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        print("loss:", val_loss / val_steps, "accuracy:",  correct / total)

    logger.info("Finished Training")


def main(config):
    assert config["dataset"] in ("librispeech", "commonvoice")
    train_model(config)


if __name__ == "__main__":
    global logger
    logger = set_logger()

    config = {"batch_size": 32, "epochs": 10, "learning_rate": 0.001, "l1": 128, "l2": 64, "dataset": "librispeech"}

    # for dataset_name in ["librispeech", "commonvoice"]:
    #     config["dataset"] = dataset_name
    #     main(config=config)

    main(config=config)
