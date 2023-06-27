import librosa
import numpy as np
import os
import pandas as pd
import speech_recognition as sr
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans


# Function to open an audio file and transcribe it.
def SpeechRecognition(language='English', file='demo1.wav'):
    language_mapping = {
        'English': 'en-US',
        'Spanish': 'es',
        'French': 'fr'
    }

    r = sr.Recognizer()

    audio_file = os.path.join('audio_files', file)  # Read from "Audios" folder
    print(f"File: {audio_file}")

    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)  # Read the complete audio

    # Transcription:
    try:
        text = r.recognize_google(audio, language=language_mapping[language])
        return audio, text
    except sr.UnknownValueError:
        print("Speech recognition could not understand audio")
        return audio, None
    except sr.RequestError as e:
        print("Error in the request: {0}".format(e))
        return audio, None


def extract_features(audio):
    # Convert audio data to a NumPy array of integers
    audio_data = np.frombuffer(audio.frame_data, dtype=np.int16)

    # Convert the data type to floating-point
    audio_data = audio_data.astype(np.float32)

    sample_rate = audio.sample_rate

    # Calculate energy features
    energy = librosa.feature.rms(y=audio_data)
    energy = np.squeeze(energy)

    # Calculate pauses
    pauses = librosa.effects.split(audio_data)
    pause_durations = librosa.frames_to_time([pause[1] - pause[0] for pause in pauses], sr=sample_rate)

    # Calculate fundamental frequency (F0)
    f0, voiced_flag = librosa.piptrack(y=audio_data, sr=sample_rate)
    f0_mean = librosa.feature.rms(y=f0)
    f0_mean = f0_mean.flatten()

    # Perform feature extraction using desired technique (e.g., MFCC)
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate)

    # Concatenate all the features into a single array
    features = np.concatenate([energy, pause_durations, f0_mean, mfcc.flatten()])

    # Return the feature array
    return features


file = 'demo1.wav'
# file = "Alphabet 2023 Q1 Earnings Call-1-minute.wav"
# file = "Alphabet 2023 Q1 Earnings Call-2-minutes.wav"
# file = "Alphabet 2023 Q1 Earnings Call-3-minutes.wav"

audio, text = SpeechRecognition(language='English', file=file)  # Read from audio file and transcribe to text
print("=" * 80)
print("Audio Transcription: \n", text)

audio_features = extract_features(audio)  # Extract features from audio
print("=" * 80)
print("Audio characteristics to save in a dataframe: \n", audio_features)
print("Audio features size: ", audio_features.shape)

# Check the words and the frequency of each one:

words = text.lower().split()

word_counts = {}

for word in words:
    if word in word_counts:
        word_counts[word] += 1
    else:
        word_counts[word] = 1

# Sort
# sorted_words = sorted(word_counts.keys())
# for word in sorted_words:
#    print(f"{word}: {word_counts[word]}")

sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
for word, count in sorted_word_counts:
    print(f"{word}: {count}")

# Convert audio_features to a dataframe
df = pd.DataFrame(audio_features.reshape(1, -1))

# Perform K-means clustering
kmeans = KMeans(n_clusters=3)  # Choose the desired number of clusters
cluster_labels = kmeans.fit_predict(audio_features.reshape(-1, 1))

# Print the cluster labels for each feature
for feature, label in zip(audio_features, cluster_labels):
    print(f"Feature: {feature}, Cluster Label: {label}")

# Create a list of tuples containing the word and its corresponding feature
word_feature_mapping = list(zip(words, audio_features))

sorted_word_feature_mapping = sorted(word_feature_mapping, key=lambda x: x[1])

for word, feature in sorted_word_feature_mapping:
    print(f"Word: {word}, Feature: {feature}")
