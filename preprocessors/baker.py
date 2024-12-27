# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
from tqdm import tqdm
import os
import torchaudio
from utils import audio
import csv
import random

from utils.util import has_existed
from text import _clean_text
import librosa
import soundfile as sf
from scipy.io import wavfile

from pathlib import Path
import numpy as np
from glob import glob


def get_lines(file):
    lines = []
    with open(file, encoding="utf-8") as f:
        for line in tqdm(f):
            lines.append(line.strip())
    return lines[::2]


def get_uid2utt(baker_path, dataset):
    index_count = 0
    total_duration = 0

    uid2utt = []
    for l in tqdm(dataset):
        items = l.split("\t")
        uid = items[0]
        text = items[1]

        res = {
            "Dataset": "baker",
            "index": index_count,
            "Singer": "baker",
            "Uid": uid,
            "Text": text,
        }

        # Duration in wav files
        audio_file = os.path.join(baker_path, "Wave/{}.wav".format(uid))

        res["Path"] = audio_file

        waveform, sample_rate = torchaudio.load(audio_file)
        duration = waveform.size(-1) / sample_rate
        res["Duration"] = duration

        uid2utt.append(res)

        index_count = index_count + 1
        total_duration += duration

    return uid2utt, total_duration / 3600


def split_dataset(
    lines, test_rate=0.05, valid_rate=0.05, test_size=None, valid_size=None
):
    if test_size == None:
        test_size = int(len(lines) * test_rate)
    if valid_size == None:
        valid_size = int(len(lines) * valid_rate)
    random.shuffle(lines)

    train_set = []
    test_set = []
    valid_set = []

    for line in lines[:test_size]:
        test_set.append(line)
    for line in lines[test_size : test_size + valid_size]:
        valid_set.append(line)
    for line in lines[test_size + valid_size :]:
        train_set.append(line)
    return train_set, test_set, valid_set


max_wav_value = 32768.0


def main(output_path, dataset_path):
    print("-" * 10)
    print("Dataset splits for {}...\n".format("Baker"))

    dataset = "baker"

    save_dir = os.path.join(output_path, dataset)
    os.makedirs(save_dir, exist_ok=True)
    baker_path = dataset_path

    train_output_file = os.path.join(save_dir, "train.json")
    test_output_file = os.path.join(save_dir, "test.json")
    valid_output_file = os.path.join(save_dir, "valid.json")
    singer_dict_file = os.path.join(save_dir, "singers.json")

    speaker = "baker"
    speakers = [dataset + "_" + speaker]
    singer_lut = {name: i for i, name in enumerate(sorted(speakers))}
    with open(singer_dict_file, "w") as f:
        json.dump(singer_lut, f, indent=4, ensure_ascii=False)

    if (
        has_existed(train_output_file)
        and has_existed(test_output_file)
        and has_existed(valid_output_file)
    ):
        return

    meta_file = os.path.join(baker_path, "ProsodyLabeling/000001-010000.txt")
    lines = get_lines(meta_file)

    train_set, test_set, valid_set = split_dataset(lines)

    res, hours = get_uid2utt(baker_path, train_set)

    # Save train
    os.makedirs(save_dir, exist_ok=True)
    with open(train_output_file, "w") as f:
        json.dump(res, f, indent=4, ensure_ascii=False)

    print("Train_hours= {}".format(hours))

    res, hours = get_uid2utt(baker_path, test_set)

    # Save test
    os.makedirs(save_dir, exist_ok=True)
    with open(test_output_file, "w") as f:
        json.dump(res, f, indent=4, ensure_ascii=False)

    print("Test_hours= {}".format(hours))

    res, hours = get_uid2utt(baker_path, valid_set)
    # Save valid
    os.makedirs(save_dir, exist_ok=True)
    with open(valid_output_file, "w") as f:
        json.dump(res, f, indent=4, ensure_ascii=False)

    print("Valid_hours= {}".format(hours))
