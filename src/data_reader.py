import os
import json
import sys
from typing import List
from tqdm import tqdm
import random
import numpy as np
import librosa
import soundfile
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(
        self,
        data_list_path,
        processor,
        mono=True,
        language=None,
        timestamps=False,
        sample_rate=16000,
        min_duration=0.5,
        max_duration=30,
        augment_config_path=None,
    ):
        """
        Args:
            data_list_path: 数据列表文件的路径，或者二进制列表的头文件路径
            processor: Whisper的预处理工具，WhisperProcessor.from_pretrained获取
            mono: 是否将音频转换成单通道，这个必须是True
            language: 微调数据的语言
            timestamps: 微调时是否使用时间戳
            sample_rate: 音频的采样率，默认是16000
            min_duration: 小于这个时间段的音频将被截断，单位秒，不能小于0.5，默认0.5s
            max_duration: 大于这个时间段的音频将被截断，单位秒，不能大于30，默认30s
            augment_config_path: 数据增强配置参数文件路径
        """
        super(CustomDataset, self).__init__()
        assert (
            min_duration >= 0.5
        ), f"min_duration must large than 0.5, failed with {min_duration}"
        assert (
            max_duration <= 30
        ), f"max_duration must smaller than 30, failed with {max_duration}"
        self.data_list_path = data_list_path
        self.processor = processor
        self.sample_rate = sample_rate
        self.mono = mono
        self.language = language
        self.timestamps = timestamps
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.vocab = self.processor.tokenizer.get_vocab()
        self.startoftranscript = self.vocab["<|startoftranscript|>"]
        self.endoftext = self.vocab["<|endoftext|>"]
        if "<|nospeech|>" in self.vocab.keys():
            self.nospeech = self.vocab["<|nospeech|>"]
            self.timestamp_begin = None
        else:
            # compactible with old model
            self.nospeech = self.vocab["<|nocaptions|>"]
            self.timestamp_begin = self.vocab["<|notimestamps|>"] + 1
        self.data_list: List[dict] = []
        self._load_data_list()

    def _load_data_list(self):
        with open(self.data_list_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.data_list = []
        for line in tqdm(lines, desc="loading data list"):
            if isinstance(line, str):
                line = json.loads(line)
            if not isinstance(line, dict):
                continue
            if line["duration"] < self.min_duration:
                continue
            if self.max_duration != -1 and line["duration"] > self.max_duration:
                continue
            self.data_list.append(dict(line))

    @staticmethod
    def resample(sample, orig_sr, target_sr):
        sample = librosa.resample(sample, orig_sr=orig_sr, target_sr=target_sr)
        return sample

    def _get_list_data(self, idx):
        data_list = self.data_list[idx]
        # get data
        chosen = data_list["chosens"] if self.timestamps else data_list["chosen"]
        rejected = data_list["rejecteds"] if self.timestamps else data_list["rejected"]
        language = data_list["language"] if "language" in data_list.keys() else None
        sample, sample_rate = soundfile.read(audio_file, dtype="float32")
        sample = sample.T
        # to mono
        if self.mono:
            sample = librosa.to_mono(sample)
        # resample
        if self.sample_rate != sample_rate:
            sample = self.resample(
                sample, orig_sr=sample_rate, target_sr=self.sample_rate
            )
        return (
            sample,
            sample_rate,
            chosen,
            rejected,
            language,
        )

    def get_audio_info(self, idx):
        data_list = self.data_list[idx]
        audio_file = data_list["audio"]["path"]
        duration = data_list["duration"]
        return audio_file, duration

    def __getitem__(self, idx):
        try:
            # step 1: get audio data
            sample, sample_rate, chosen, rejected, language = self._get_list_data(
                idx=idx
            )
            # set up language
            self.processor.tokenizer.set_prefix_tokens(
                language=language if language is not None else self.language
            )
            # step 2: get feature of log-Mel
            feature_log_mel = self.processor(
                audio=sample, sampling_rate=self.sample_rate
            )
            # step 3: get the positive and negative label
            positive_input_ids = self.processor.tokenizer(chosen)
            negative_input_ids = self.processor.tokenizer(rejected)
            data = {
                "input_features": feature_log_mel["input_features"],
                "positive_input_ids": positive_input_ids["input_ids"],
                "negative_input_ids": negative_input_ids["input_ids"],
            }
            return data
        except Exception as e:
            print(
                f"error while loading data, whose index is {idx}. error info: {e}",
                file=sys.stderr,
            )
            return self.__getitem__(random.randint(0, self.__len__() - 1))

    def __len__(self):
        return len(self.data_list)
