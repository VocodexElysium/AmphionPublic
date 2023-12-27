# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random

import numpy as np

from torch.nn import functional as F

from torch.nn.utils.rnn import pad_sequence
from utils.data_utils import *
from utils.dsp import *
from models.vocoders.vocoder_dataset import VocoderDataset


class AutoregressiveVocoderDataset(VocoderDataset):
    def __init__(self, cfg, dataset, is_valid=False):
        """
        Args:
            cfg: config
            dataset: dataset name
            is_valid: whether to use train or valid dataset
        """
        super().__init__(cfg, dataset, is_valid)

        eval_index = random.randint(0, len(self.metadata) - 1)
        eval_utt_info = self.metadata[eval_index]
        eval_utt = "{}_{}".format(eval_utt_info["Dataset"], eval_utt_info["Uid"])
        self.eval_audio = np.load(self.utt2audio_path[eval_utt])
        if cfg.preprocess.use_mel:
            self.eval_mel = np.load(self.utt2mel_path[eval_utt])
        if cfg.preprocess.use_frame_pitch:
            self.eval_pitch = np.load(self.utt2frame_pitch_path[eval_utt])
        

    def __getitem__(self, index):
        utt_info = self.metadata[index]

        dataset = utt_info["Dataset"]
        uid = utt_info["Uid"]
        utt = "{}_{}".format(dataset, uid)

        single_feature = dict()

        if self.cfg.preprocess.use_mel:
            mel = np.load(self.utt2mel_path[utt])
            assert mel.shape[0] == self.cfg.preprocess.n_mel

            if "target_len" not in single_feature.keys():
                single_feature["target_len"] = mel.shape[1]

            mel_win = self.cfg.preprocess.cut_mel_frame + 2 * self.cfg.preprocess.mel_frame_pad

            if single_feature["target_len"] <= 2 + mel_win + 2 * self.cfg.preprocess.mel_frame_pad:
                if single_feature["target_len"] <= mel_win:
                    mel = np.pad(
                        mel,
                        ((0, 0), (0, mel_win - mel.shape[-1])),
                        mode="constant",
                    )
                mel = mel[:, 0 : mel_win]
            else:
                if "start" not in single_feature.keys():
                    start = random.randint(
                        0, mel.shape[-1] - 2 - (mel_win + 2 * self.cfg.preprocess.mel_frame_pad)
                    )
                    end = start + mel_win
                    single_feature["start"] = start
                    single_feature["end"] = end
                mel = mel[:, single_feature["start"] : single_feature["end"]]
            single_feature["mel"] = mel

        if self.cfg.preprocess.use_audio:
            audio = np.load(self.utt2audio_path[utt])

            if self.cfg.preprocess.audio_mode == "mu_law" or self.cfg.preprocess.audio_mode == "mu_law_one_hot":
                audio = compress(audio, bits=self.cfg.preprocess.bits)
            elif self.cfg.preprocess.audio_mode == "mu_law_quantize" or self.cfg.preprocess.audio_mode == "mu_law_quantize_one_hot":
                audio = compress(audio, bits=self.cfg.preprocess.bits)
                audio = audio_to_label(audio, bits = self.cfg.preprocess.bits)
            else:
                raise NotImplementedError

            assert "target_len" in single_feature.keys()

            if single_feature["target_len"] <= 2 + mel_win + 2 * self.cfg.preprocess.mel_frame_pad:
                audio = np.pad(
                    audio,
                    (
                        (
                            0,
                            mel_win 
                            * self.cfg.preprocess.hop_size
                            - audio.shape[-1],
                        )
                    ),
                    mode="constant",
                )
                audio = audio[self.cfg.preprocess.mel_frame_pad * self.cfg.preprocess.hop_size : (self.cfg.preprocess.mel_frame_pad + self.cfg.preprocess.cut_mel_frame) * self.cfg.preprocess.hop_size]
            else:
                if "start" not in single_feature.keys():
                    audio = audio[self.cfg.preprocess.mel_frame_pad * self.cfg.preprocess.hop_size : (self.cfg.preprocess.mel_frame_pad + self.cfg.preprocess.cut_mel_frame) * self.cfg.preprocess.hop_size]
                else:
                    audio = audio[
                        (single_feature["start"] + self.cfg.preprocess.mel_frame_pad)
                        * self.cfg.preprocess.hop_size : (single_feature["start"] + self.cfg.preprocess.mel_frame_pad + self.cfg.preprocess.cut_mel_frame)
                        * self.cfg.preprocess.hop_size,
                    ]
            single_feature["audio"] = audio

            if self.cfg.preprocess.audio_mode == "mu_law":
                audio_prev = audio[: self.cfg.preprocess.cut_mel_frame * self.cfg.preprocess.hop_size]
                audio_next = audio[1:]
            elif self.cfg.preprocess.audio_mode == "mu_law_quantize":
                audio_prev = audio[: self.cfg.preprocess.cut_mel_frame * self.cfg.preprocess.hop_size]
                audio_next = audio[1:]
                audio_prev = label_to_audio(audio_prev.float(), self.cfg.preprocess.bits)
            elif self.cfg.preprocess.audio_mode == "mu_law_one_hot":
                audio_prev = audio.reshape(-1, 1)
                audio_next = audio_prev.copy()
                audio_prev = audio_prev.float().transpose(0, 1).contiguous()
                audio_next = audio_next.float().unsqueeze(-1).contiguous()
            elif self.cfg.preprocess.audio_mode == "mu_law_quantize_one_hot":
                audio_prev = label_to_onehot(audio, self.cfg.preprocess.bits)
                audio_next = audio_prev.copy()
                audio_prev = audio_prev.float().transpose(0, 1).contiguous()
                audio_next = audio_next.float().unsqueeze(-1).contiguous()
            
            single_feature["audio_prev"] = audio_prev
            single_feature["audio_next"] = audio_next

        return single_feature


class AutoregressiveVocoderCollator(object):
    """Zero-pads model inputs and targets based on number of frames per step"""

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, batch):
        packed_batch_features = dict()

        # mel: [b, n_mels, frame]
        # frame_pitch: [b, frame]
        # audios: [b, frame * hop_size]

        for key in batch[0].keys():
            if key in ["target_len", "start", "end"]:
                continue
            else:
                values = [torch.from_numpy(b[key]) for b in batch]
                packed_batch_features[key] = pad_sequence(
                    values, batch_first=True, padding_value=0
                )

        return packed_batch_features
