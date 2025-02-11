import re
from typing import Any, List, Dict, Union
from zhconv import convert
from dataclasses import dataclass
import torch


def remove_punctuation(text: str or List[str]):
    punctuation = "!,.;:?、！，。；：？"
    if isinstance(text, str):
        text = re.sub(r"[{}]+".format(punctuation), "", text).strip()
        return text
    elif isinstance(text, list):
        result_text = []
        for t in text:
            t = re.sub(r"[{}]+".format(punctuation), "", t).strip()
            result_text.append(t)
        return result_text
    else:
        raise Exception(f"do not support type of {type(text)}")


def to_simple(text: str or List[str]):
    if isinstance(text, str):
        text = convert(text, "zh-cn")
        return text
    elif isinstance(text, list):
        result_text = []
        for t in text:
            t = convert(t, "zh-cn")
            result_text.append(t)
        return result_text
    else:
        raise Exception(f"do not support type of {type(text)}")


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __label_padding(self, labels):
        # pad the labels to max length
        batch = self.processor.tokenizer.pad(labels, return_tensors="pt")
        # replace padding with -100 to ignore loss correctly
        input_ids = batch["input_ids"].masked_fill(batch.attention_mask.ne(1), -100)
        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (
            (input_ids[:, 0] == self.processor.tokenizer.bos_token_id)
            .all()
            .cpu()
            .item()
        ):
            input_ids = input_ids[:, 1:]
        return input_ids

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # step 1: treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        # step 2: padding positive label sequences
        batch["positive_input_ids"] = self.__label_padding(
            [{"input_ids": feature["positive_input_ids"]} for feature in features]
        )
        # step 3: padding negative label sequences
        batch["negative_input_ids"] = self.__label_padding(
            [{"input_ids": feature["negative_input_ids"]} for feature in features]
        )
        return batch
