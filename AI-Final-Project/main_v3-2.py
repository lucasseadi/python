# loads the datasets and the pretrained nn_models with train test split
# no train, just validation loop
# TranscriptionNet

import os
import random
from torch.utils.data import Dataset

from nn_models.CNN import CNN
from nn_models.DiffusionModel import SoundAnalysisModel
from MyCommonVoice import MyCommonVoice
from nn_models.TranscriptionNet import TranscriptionNet
from utils import *

vocabulary = None


def load_data(dataset_name):
    logger.info(f"Loading {dataset_name} dataset...")
    train_set, val_set, test_set = Dataset(), Dataset(), Dataset()
    if dataset_name == "librispeech":
        max_audio_length = 392400
        max_transcript_length = 398
        dataset_path = os.path.dirname(os.path.abspath(__file__)) + "/data/librispeech/"
        char2index = {'D': 0, 'G': 1, 'O': 2, 'S': 3, 'J': 4, 'T': 5, ' ': 6, 'E': 7, 'X': 8, 'A': 9, 'K': 10, 'R': 11,
                      'N': 12, 'L': 13, 'U': 14, 'C': 15, 'I': 16, 'M': 17, "'": 18, 'H': 19, 'Q': 20, 'B': 21, 'Y': 22,
                      'Z': 23, 'W': 24, 'F': 25, 'P': 26, 'V': 27}

        # randomly defines which entries will go to train_set and test_set
        # librispeech train-clean-100 has 28539 entries
        # 28539 * 0.02 = 570
        train_test_lines = list(range(1, 28540))

        global vocabulary
        vocabulary = tokenize_transcripts_librispeech(config, dataset_path, train_test_lines, char2index)

        random.shuffle(train_test_lines)
        train_lines = train_test_lines[570:]
        test_lines = train_test_lines[:570]

        train_set = MyLibriSpeech(dataset_path, url="train-clean-100", download=True, dataset_lines=train_lines,
                                  char2index=char2index)
        test_set = MyLibriSpeech(dataset_path, url="train-clean-100", download=True, dataset_lines=test_lines,
                                char2index=char2index)
        # test_set = MyLibriSpeech(dataset_path, url="test-other", download=True, char2index=char2index)

        print("train_set", len(train_set))
        print("test_set", len(test_set))

        train_set.max_audio_length = max_audio_length
        train_set.max_transcript_length = max_transcript_length

        test_set.max_audio_length = max_audio_length
        test_set.max_transcript_length = max_transcript_length

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

    return train_set, val_set, test_set, max_audio_length


def load_state_librispeech(config, model, max_audio_length):
    logger.info(f"Loading librispeech state...")
    checkpoint_path = os.path.dirname(os.path.abspath(__file__)) + "/data/librispeech_pretrained.pth"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model_state_dict = checkpoint["state_dict"]
    global vocabulary
    logger.info(f"Initializing {model} model...")
    if model == "CNN":
        model = CNN(config, len(vocabulary))
    elif model == "TranscriptionNet":
        model = TranscriptionNet(config, vocabulary, max_audio_length)
    elif model == "DiffusionModel":
        model = SoundAnalysisModel(config, vocabulary, max_audio_length)
    logger.info(f"Loading state to librispeech model...")
    model.load_state_dict(model_state_dict, strict=False)
    return model


def load_state_commonvoice(config):
    logger.info("[load_state_commonvoice]")
    from speechbrain.pretrained import EncoderDecoderASR
    model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-wav2vec2-commonvoice-en",
                                           savedir="pretrained_models/asr-wav2vec2-commonvoice-en")
    return model


def load_model(config, model, max_audio_length):
    logger.info(f"Loading {config['dataset']} state...")
    if config["dataset"] == "librispeech":
        model = load_state_librispeech(config, model, max_audio_length)
    elif config["dataset"] == "commonvoice":
        model = load_state_commonvoice(config)
    return model


# data_dir arg is necessary for trial to work
def test_accuracy(config, model_name="TranscriptionNet", data_dir=None):
    ########## Loads data ##########
    train_set, val_set, test_set, max_audio_length = load_data(config["dataset"])
    # train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    # val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False)

    ########## Loads pre-trained model ##########
    model = load_model(config, model_name, max_audio_length)

    logger.info(f"Will evaluate pre-trained model...")

    ########## Evaluates pre-trained model ##########
    correct = 0
    total = 0
    with torch.no_grad():
        total_sentences = 0
        correct_sentences = 0
        for batch in tqdm(test_loader, desc=f"Testing {model_name} model accuracy", unit="batch"):
            audios, transcripts = batch

            # Add channel dimension for the NN model
            audios = audios.unsqueeze(1)

            # Get the spectrogram for the audio
            spectrograms = get_spectrogram(audios)

            outputs = model(spectrograms)
            print("[test_accuracy] outputs.data", outputs.data.shape)
            _, predicted = torch.max(outputs.data, 1)
            print("[test_accuracy] predicted", predicted.shape)
            predicted_list = predicted.tolist()
            print("[test_accuracy] predicted_list", len(predicted_list))
            global vocabulary
            predicted = convert_to_sentence(predicted, vocabulary)
            break

            print("[test_accuracy] predicted", predicted)
            print("[test_accuracy] transcripts", transcripts)

            # Compare predicted and ground truth transcripts
            for pred, target in zip(predicted, transcripts):
                if pred.strip() == target.strip():
                    correct += 1
                total += 1

    if correct_sentences > 0:
        accuracy = correct_sentences / total_sentences
    else:
        accuracy = 0
    print("correct_sentences", correct_sentences)
    print("total_sentences", total_sentences)
    print("Pre-trained model accuracy:", accuracy)


def main(config):
    assert config["dataset"] in ("librispeech", "commonvoice")
    # test_accuracy(config, "CNN")
    # test_accuracy(config, "TranscriptionNet")
    test_accuracy(config, "DiffusionModel")


if __name__ == "__main__":
    global logger
    logger = set_logger()

    config = {"batch_size": 16, "epochs": 10, "learning_rate": 0.001, "l1": 128, "l2": 64, "dataset": "librispeech"}

    # for dataset_name in ["librispeech", "commonvoice"]:
    #     config["dataset"] = dataset_name
    #     main(config=config)

    main(config=config)
