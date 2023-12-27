# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This model code is adopted under the MIT License
# https://github.com/r9y9/wavenet_vocoder

import torch
import numpy as np
import time
import torch.nn as nn
import math

from tqdm import tqdm

from utils.distribution import (
    sample_from_discretized_mix_logistic,
    sample_from_mix_gaussian,
)
from utils.dsp import (
    decompress,
    label_to_audio,
)
import torch.nn.functional as F


def wavenet_inference(
    cfg, model, initial_input, mel, T, softmax, quantize, log_scale_min, device
):
    """Generation

    Due to linearized convolutions, inputs of shape (B x C x T) are reshaped
    to (B x T x C) internally and fed to the network for each time step.
    Input of each time step will be of shape (B x 1 x C).

    Args:
        initial_input (Tensor): Initial decoder input, (B x C x 1)
        mel (Tensor): Local conditioning features, shape (B x C' x T)
        T (int): Number of time steps to generate.
        tqdm (lamda) : tqdm
        softmax (bool) : Whether applies softmax or not
        quantize (bool): Whether quantize softmax output before feeding the
        network output to input for the next time step.
        log_scale_min (float):  Log scale minimum value.

    Returns:
        Tensor: Generated one-hot encoded samples. B x C x T
          or scaler vector B x 1 x T
    """
    model.clear_buffer()

    T = int(T)
    B = mel.shape[0]

    mel = model.upsample_net(mel)
    assert mel.size(-1) == T
    mel = mel.transpose(1, 2).contiguous()

    outputs = []

    if initial_input is None:
        if model.cfg.VOCODER.SCALAR_INPUT:
            initial_input = torch.zeros(B, 1, 1)
        else:
            initial_input = torch.zeros(B, 1, model.out_channels)
            initial_input[:, :, 127] = 1
    else:
        if initial_input.size(1) == model.out_channels:
            initial_input = initial_input.transpose(1, 2).contiguous()

    initial_input = initial_input.to(device)

    current_input = initial_input

    for t in tqdm(range(T)):
        if t > 0:
            current_input = outputs[-1]

        # Conditioning features for single time step
        melt = mel[:, t, :].unsqueeze(1)

        x = current_input
        x = model.first_conv.incremental_forward(x)

        skips = 0
        for f in model.conv_layers:
            x, h = f.incremental_forward(x, melt)
            skips += h
        skips *= math.sqrt(1.0 / len(model.conv_layers))

        x = skips
        for f in model.last_conv_layers:
            try:
                x = f.incremental_forward(x)
            except AttributeError:
                x = f(x)

        # Generate next input by sampling
        if model.cfg.VOCODER.SCALAR_INPUT:
            x = sample_from_discretized_mix_logistic(
                x.view(B, -1, 1), log_scale_min=log_scale_min
            )
        else:
            x = F.softmax(x.view(B, -1), dim=1) if softmax else x.view(B, -1)
            if quantize:
                dist = torch.distributions.OneHotCategorical(x)
                x = dist.sample()
        outputs += [x.data]

    # T x B x C
    outputs = torch.stack(outputs)
    # B x C x T
    outputs = outputs.transpose(0, 1).transpose(1, 2).contiguous()

    model.clear_buffer()
    return outputs.detach().cpu()