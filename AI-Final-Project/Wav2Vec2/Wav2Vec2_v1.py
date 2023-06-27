# The large model pretrained and fine-tuned on 960 hours of Librispeech on 16kHz sampled speech audio.
# When using the model make sure that your speech input is also sampled at 16Khz.

# Some weights of Wav2Vec2ForCTC were not initialized from the model checkpoint at facebook/wav2vec2-large-960h and are
# newly initialized: ['wav2vec2.masked_spec_embed']
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

import torch
import torchaudio
from torchaudio.transforms import Resample
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


# Load the pretrained model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# Set the model in evaluation mode
model.eval()


# Define a function to transcribe the audio
def transcribe_audio(audio_path):
    # Load the audio waveform
    waveform, sample_rate = torchaudio.load(audio_path)

    # Resample the waveform if necessary
    if sample_rate != 16000:
        resampler = Resample(sample_rate, 16000)
        waveform = resampler(waveform)

    # Preprocess the audio waveform
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)

    # Forward pass through the model
    with torch.no_grad():
        logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

    # Decode the transcription from the model output
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    return transcription


# Example usage
audio_path = "audio_files/demo1.wav"
transcription = transcribe_audio(audio_path)
print("Transcription:", transcription)
