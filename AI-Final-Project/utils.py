import logging
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from tqdm import tqdm

from MyLibriSpeech import MyLibriSpeech


def convert_to_sentence(predicted_indices, vocabulary):
    sentence = []
    print("[convert_to_sentence] predicted_indices", predicted_indices)
    # _, predicted_indices = torch.max(output, dim=2)

    # for index in predicted_indices[0]:
    for index in predicted_indices:
        # token = vocabulary[index.item()]
        token = vocabulary[index]
        if token == "<eos>":
            break
        sentence.append(token)

    return (" ".join(sentence),)


# Finds the maximum length among all audio tensors
def find_max_audio_length(dataset, type="flac"):
    if type == "flac":
        return max(audio.size(1) for audio, transcript in tqdm(dataset, desc="Finding max audio length", unit="entry"))
    if type == "mp3":
        return max(audio.size() for audio, transcript in tqdm(dataset, desc="Finding max audio length", unit="entry"))


# Finds the maximum length among all audio tensors
def find_max_transcript_length(dataset, type="flac"):
    return max(len(transcript) for audio, transcript in tqdm(dataset, desc="Finding max transcript length", unit="entry"))


def get_spectrogram(audios):
    # print("[get_spectrogram] waveform", audios.shape)
    spectrograms = []
    for waveform in audios:
        # Convert the waveform to mono if it's stereo
        waveform_mono = waveform.mean(dim=0)

        # Apply the spectrogram transformation
        transform = torchaudio.transforms.Spectrogram()
        spectrogram = transform(waveform_mono)

        # Normalize the spectrogram values for better model performance
        spectrogram = torch.log1p(spectrogram)

        # Reshape the spectrogram tensor to match the expected input shape
        # spectrogram = spectrogram.unsqueeze(0)

        spectrograms.append(spectrogram)

    spectrograms = torch.stack(spectrograms)
    # print("[get_spectrogram] spectrograms", spectrograms.shape)

    return spectrograms


def set_char2index(dataset):
    transcripts = []
    for sound, transcript in tqdm(dataset, desc="Setting char2index", unit="entry"):
        transcripts.append(transcript)

    # Create a set of unique characters in the transcripts
    unique_chars = set(''.join(transcripts))

    # Create the char2index mapping
    return {char: index for index, char in enumerate(unique_chars)}


def set_logger():
    logging.basicConfig(filename="log.txt", filemode="w",
                        format="[%(asctime)s %(filename)s %(funcName)s %(levelname)s line %(lineno)d] - %(message)s",
                        datefmt="%Y/%m/%d %H:%M:%S",
                        level=logging.DEBUG)
    logger = logging.getLogger("FinalProject")
    return logger

"""
def tokenize_texts(transcript):
    tokenizer = get_tokenizer('basic_english')

    # Tokenize each string in the transcript
    tokenized_transcript = [tokenizer(text) for text in transcript]

    # Create a vocabulary (a set of unique words) from the tokenized transcript
    vocab = set()
    for tokens in tokenized_transcript:
        vocab.update(tokens)

    # Add the <pad> token to the vocabulary
    vocab.add('<pad>')

    # Create a mapping from words to unique numeric identifiers
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}

    # Convert each tokenized string into a sequence of numeric identifiers
    indexed_transcript = [[word_to_idx[word] for word in tokens] for tokens in tokenized_transcript]

    # Pad the sequences to a maximum length
    padded_transcript = pad_sequence([torch.tensor(seq) for seq in indexed_transcript], batch_first=True,
                                     padding_value=word_to_idx['<pad>'])

    return padded_transcript
"""


def tokenize_transcripts_librispeech(config, dataset_path, train_test_lines, char2index):
    dataset = MyLibriSpeech(dataset_path, url="train-clean-100", download=True, dataset_lines=train_test_lines,
                            char2index=char2index)
    dataset_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    tokens = []
    for batch in tqdm(dataset_loader, desc="Tokenizing dataset", unit="batch"):
        audios, transcripts = batch
        for transcript in transcripts:
            tokens.extend(transcript.split())

    # Create vocabulary
    vocabulary = list(set(tokens))

    print("[tokenize_transcripts_librispeech] num_classes", len(vocabulary))

    # Define vocabulary size
    # vocabulary_size = len(vocabulary)
    # print("[tokenize_transcripts_librispeech] Vocabulary size:", vocabulary_size)

    return vocabulary
