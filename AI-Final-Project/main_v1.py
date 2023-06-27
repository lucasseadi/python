# loads the datasets and trains them

import torch
import torch.nn as nn
from nn_models.CNN import CNN
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import find_max_audio_length, get_spectrogram, set_char2index
from MyLibriSpeech import MyCommonVoice, MyLibriSpeech


def train(config):
    ########## Loads data ##########
    print(f"Loading {config['dataset']} dataset...")
    dataset = Dataset()
    if config["dataset"] == "librispeech":
        dataset_path = 'data/librispeech/'
        dataset = MyLibriSpeech(dataset_path, url="train-clean-100", download=True)
        max_length = 392400
        dataset.char2index = {'D': 0, 'G': 1, 'O': 2, 'S': 3, 'J': 4, 'T': 5, ' ': 6, 'E': 7, 'X': 8, 'A': 9, 'K': 10,
                              'R': 11, 'N': 12, 'L': 13, 'U': 14, 'C': 15, 'I': 16, 'M': 17, "'": 18, 'H': 19, 'Q': 20,
                              'B': 21, 'Y': 22, 'Z': 23, 'W': 24, 'F': 25, 'P': 26, 'V': 27}

    elif config["dataset"] == "commonvoice":
        dataset_path = 'data/cv-corpus-13.0-delta-2023-03-09/'
        dataset = MyCommonVoice(dataset_path)
        max_length = 352512
        dataset.char2index = set_char2index(dataset)
    else:
        # Finds the max length so __getItem__ can do padding
        dataset.padding = False
        max_length = find_max_audio_length(dataset)
        dataset.padding = True
        print("max_length", max_length)
        dataset.char2index = set_char2index(dataset)
    dataset.max_length = max_length
    data_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    ########## Loads model ##########
    model = CNN(config)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    for epoch in range(config["epochs"]):
        model.train()

        loss = 0.0

        # Iterate over the data loader
        for batch in tqdm(data_loader, desc="Training model", unit="batch"):
            audio, transcript = batch

            print("[train] audio", audio.shape)

            # Add channel dimension for the CNN model
            audio = audio.unsqueeze(1)

            print("[train] audio", audio.shape)

            # Get the spectrogram for the audio
            spectrogram = get_spectrogram(audio)

            print("[train] spectrogram", spectrogram.shape)

            # Clear the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(spectrogram)

            # Adjust the transcript batch size
            transcript = transcript[:outputs.size(0)]  # Adjust batch size to match outputs
            # print("[train] transcript 1", transcript)
            transcript = ''.join(transcript)
            # print("[train] transcript 2", transcript)

            # Convert the transcript to a tensor of Long type
            transcript = [dataset.char2index[c] for c in transcript]
            # print("[train] transcript 3", transcript)
            transcript = torch.tensor(transcript, dtype=torch.long)
            # print("[train] transcript 4", transcript.shape)
            # print("[train] outputs", outputs.shape)

            # Compute the CTC loss
            # loss = criterion(outputs.permute(2, 0, 1), transcript, input_lengths=[outputs.size(2)],
            #                  target_lengths=[len(transcript)])
            # loss = nn.CTCLoss()(outputs, transcript, input_lengths=torch.tensor([outputs.shape[0]]),
            #                     target_lengths=torch.tensor([transcript.shape[0]]))

            # Backward pass
            # loss.backward()

            # Update the weights
            optimizer.step()

            # loss += loss.item() * audio.size(0)

        epoch_loss = loss / len(dataset)
        print(f"[Epoch {epoch + 1}/{config['epochs']}] Loss: {epoch_loss}")


def load_state_librispeech():
    print("[load_state_librispeech]")
    checkpoint_path = "data/librispeech_pretrained.pth"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model_state_dict = checkpoint["state_dict"]
    # for key in model_state_dict.keys():
    #     print(key, model_state_dict[key].shape)
    model = CNN(config)
    model.load_state_dict(model_state_dict, strict=False)
    model.eval()


def load_state_commonvoice():
    print("[load_state_commonvoice]")
    from speechbrain.pretrained import EncoderDecoderASR
    asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-wav2vec2-commonvoice-en",
                                               savedir="pretrained_models/asr-wav2vec2-commonvoice-en")


if __name__ == "__main__":
    config = {
        "batch_size": 32,
        "epochs": 10,
        "learning_rate": 0.001,
        "l1": 128,
        "l2": 64
    }

    for dataset_name in ["librispeech", "commonvoice"]:
        config["dataset"] = dataset_name
        train(config=config)
