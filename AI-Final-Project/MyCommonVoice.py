import csv
import numpy as np
import os
import random
import torch
from pathlib import Path
from pydub import AudioSegment
from torch.utils.data import Dataset


class MyCommonVoice(Dataset):
    max_length = 0
    padding = True

    def __init__(self, root, train=True):
        super().__init__()

        self.tsv_root = os.path.join(root, "en")
        self.tsv_files = ["invalidated.tsv", "other.tsv", "validated.tsv"]
        self._ext_audio = ".mp3"
        self._path = os.path.join(self.tsv_root, "clips")
        self._walker = sorted(str(p.stem) for p in Path(self._path).glob("*/*/*" + self._ext_audio))
        self.val_set = []
        self.test_set = []

        if train:
            self.process_common_voice_train_set()

    # Reads the TSV files and process the data
    def process_common_voice_train_set(self):
        # randomly defines which entries will go to val_set and test_set
        val_test_lines = random.sample(range(1, 30281), 12112)
        val_lines = val_test_lines[:6056]
        test_lines = val_test_lines[6056:]

        line_number = 1
        for tsv_file in self.tsv_files:
            with open(os.path.join(self.tsv_root, tsv_file), 'r') as file:
                reader = csv.DictReader(file, delimiter='\t')

                for line in reader:
                    audio_path = line['path']
                    transcript = line['sentence']

                    if line_number in val_lines:
                        # Add the processed data to val set
                        self.val_set.append([audio_path, transcript])
                    elif line_number in test_lines:
                        # Add the processed data to test set
                        self.test_set.append([audio_path, transcript])
                    else:
                        # Add the processed data to train set
                        self._walker.append((audio_path, transcript, None, None, None, None))
                    line_number += 1

    def process_common_voice_val_set(self):
        for audio_path, transcript in self.val_set:
            self._walker.append((audio_path, transcript, None, None, None, None))

    def process_common_voice_test_set(self):
        for audio_path, transcript in self.test_set:
            self._walker.append((audio_path, transcript, None, None, None, None))

    def __getitem__(self, n):
        file = self._walker[n]
        audio = AudioSegment.from_file(os.path.join(self._path, file[0]), format="mp3")
        audio_array = np.array(audio.get_array_of_samples())
        audio_tensor = torch.from_numpy(audio_array).float()
        if self.padding:
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, self.max_length - audio_tensor.size(0)))
        label = file[1]

        return audio_tensor, label

    def __len__(self):
        return len(self._walker)
