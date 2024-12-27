# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from modules.naturalpseech2.transformers import (
    TransformerEncoder,
    DurationPredictor,
    PitchPredictor,
    LengthRegulator,
)
import utils.monotonic_align as monotonic_align #cd monotonic_align; python setup.py build_ext --inplace
import math
from new_text import symbols, num_tones

def fix_len_compatibility(length, num_downsamplings_in_unet=2):
    factor = torch.scalar_tensor(2).pow(num_downsamplings_in_unet)
    length = (length / factor).ceil() * factor
    if not torch.onnx.is_in_onnx_export():
        return length.int().item()
    else:
        return length

def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)

def convert_pad_shape(pad_shape):
    inverted_shape = pad_shape[::-1]
    pad_shape = [item for sublist in inverted_shape for item in sublist]
    return pad_shape

def generate_path(duration, mask):
    device = duration.device

    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)
    path = torch.zeros(b, t_x, t_y, dtype=mask.dtype).to(device=device)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - torch.nn.functional.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path * mask
    return path

class PriorEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.enc_emb_tokens = nn.Embedding(
            len(symbols), cfg.encoder.encoder_hidden, padding_idx=0
        )
        self.tone_emb_tokens = nn.Embedding(
            num_tones, cfg.encoder.encoder_hidden, padding_idx=0
        )
        self.enc_emb_tokens.weight.data.normal_(mean=0.0, std=1e-5)
        self.encoder = TransformerEncoder(
            enc_emb_tokens=self.enc_emb_tokens, tone_emb_tokens=self.tone_emb_tokens, cfg=cfg.encoder
        )
        self.proj_m = nn.Conv1d(128, 128, 1)

        self.duration_predictor = DurationPredictor(cfg.duration_predictor)
        self.pitch_predictor = PitchPredictor(cfg.pitch_predictor)
        self.length_regulator = LengthRegulator()

        self.pitch_min = cfg.pitch_min
        self.pitch_max = cfg.pitch_max
        self.pitch_bins_num = cfg.pitch_bins_num

        pitch_bins = torch.exp(
            torch.linspace(
                np.log(self.pitch_min), np.log(self.pitch_max), self.pitch_bins_num - 1
            )
        )
        self.register_buffer("pitch_bins", pitch_bins)

        self.pitch_embedding = nn.Embedding(
            self.pitch_bins_num, cfg.encoder.encoder_hidden
        )

    def forward(
        self,
        phone_id,
        tone_id,
        latent=None,
        duration=None,
        pitch=None,
        phone_mask=None,
        mask=None,
        ref_emb=None,
        ref_mask=None,
        is_inference=False,
    ):
        """
        input:
        phone_id: (B, N)
        duration: (B, N)
        pitch: (B, T)
        phone_mask: (B, N); mask is 0
        mask: (B, T); mask is 0
        ref_emb: (B, d, T')
        ref_mask: (B, T'); mask is 0

        output:
        prior_embedding: (B, d, T)
        pred_dur: (B, N)
        pred_pitch: (B, T)
        """

        x = self.encoder(phone_id, phone_mask, tone_id, ref_emb.transpose(1, 2))
        # print(torch.min(x), torch.max(x))
        mu_x = self.proj_m(x.transpose(1, 2)).transpose(1, 2) * phone_mask.unsqueeze(-1) 
        # m_x, logs_x = torch.split(x_stats, 512, dim=2)
        dur_pred_out = self.duration_predictor(x.detach(), phone_mask, ref_emb, ref_mask)
        # dur_pred_out: {dur_pred_log, dur_pred, dur_pred_round}
        
        


        if is_inference:
            logw = dur_pred_out["dur_pred_log"]
            w = torch.exp(logw) * phone_mask
            w_ceil = torch.ceil(w)
            y_lengths = torch.clamp_min(torch.sum(w_ceil, 1), 1).long()
            y_max_length = y_lengths.max()
            y_max_length_ = fix_len_compatibility(y_max_length)

            # Using obtained durations `w` construct alignment map `attn`
            y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(phone_mask.dtype)
            attn_mask = phone_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
            attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
        else:
            attn_mask = phone_mask.unsqueeze(1).unsqueeze(-1) * mask.unsqueeze(1).unsqueeze(2)
            y = latent
            # with torch.no_grad():
            #     # negative cross-entropy
            #     s_p_sq_r = torch.exp(-2 * logs_x)  # [b, d, t]
            #     neg_cent1 = torch.sum(
            #         -0.5 * math.log(2 * math.pi) - logs_x, [2], keepdim=True
            #     )  # [b, 1, t_s]
            #     neg_cent2 = torch.matmul(
            #         s_p_sq_r, -0.5 * (y**2)
            #     )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            #     neg_cent3 = torch.matmul(
            #         (m_x * s_p_sq_r), y
            #     )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            #     neg_cent4 = torch.sum(
            #         -0.5 * (m_x**2) * s_p_sq_r, [2], keepdim=True
            #     )  # [b, 1, t_s]
            #     neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4
            #     attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1))
            #     attn = attn.detach()
            with torch.no_grad():
                const = -0.5 * math.log(2 * math.pi) * mu_x.size(-1)
                factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
                y_square = torch.matmul(factor, y**2)
                y_mu_double = torch.matmul(2.0 * (factor * mu_x), y)
                mu_square = torch.sum(factor * (mu_x**2), 2).unsqueeze(-1)
                log_prior = y_square - y_mu_double + mu_square + const

                attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
                attn = attn.detach()  # b, t_text, T_mel
        
        logw_ = torch.log(1e-8 + torch.sum(attn, -1)) * phone_mask
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x)
        # mu_y = mu_y.transpose(1, 2)


        pitch_pred_log = self.pitch_predictor(mu_y, mask, ref_emb, ref_mask)

        if is_inference or pitch is None:
            pitch_tokens = torch.bucketize(pitch_pred_log.exp(), self.pitch_bins)
            pitch_embedding = self.pitch_embedding(pitch_tokens)
        else:
            pitch_tokens = torch.bucketize(pitch, self.pitch_bins)
            pitch_embedding = self.pitch_embedding(pitch_tokens)

        mu_y = mu_y + pitch_embedding

        if (not is_inference) and (mask is not None):
            mu_y = mu_y * mask.to(mu_y.dtype)[:, :, None]

        prior_out = {
            "dur_pred_round": dur_pred_out["dur_pred_round"],
            "dur_pred_log": dur_pred_out["dur_pred_log"],
            "dur_pred": dur_pred_out["dur_pred"],
            "pitch_pred_log": pitch_pred_log,
            "pitch_token": pitch_tokens,
            #"mel_len": mel_len,
            "prior_out": mu_y,
            "logw_": logw_,
            "attn": attn,
        }

        return prior_out

    def inference(
        self,
        phone_id,
        latent=None,
        duration=None,
        pitch=None,
        phone_mask=None,
        mask=None,
        ref_emb=None,
        ref_mask=None,
        is_inference=False,
    ):
        """
        input:
        phone_id: (B, N)
        duration: (B, N)
        pitch: (B, T)
        phone_mask: (B, N); mask is 0
        mask: (B, T); mask is 0
        ref_emb: (B, d, T')
        ref_mask: (B, T'); mask is 0

        output:
        prior_embedding: (B, d, T)
        pred_dur: (B, N)
        pred_pitch: (B, T)
        """

        x = self.encoder(phone_id, phone_mask, ref_emb.transpose(1, 2))
        # print(torch.min(x), torch.max(x))
        dur_pred_out = self.duration_predictor(x, phone_mask, ref_emb, ref_mask)
        # dur_pred_out: {dur_pred_log, dur_pred, dur_pred_round}

        if is_inference or duration is None:
            x, mel_len = self.length_regulator(
                x,
                dur_pred_out["dur_pred_round"],
                max_len=torch.max(torch.sum(dur_pred_out["dur_pred_round"], dim=1)),
            )


        pitch_pred_log = self.pitch_predictor(x, mask, ref_emb, ref_mask)

        if is_inference or pitch is None:
            pitch_tokens = torch.bucketize(pitch_pred_log.exp(), self.pitch_bins)
            pitch_embedding = self.pitch_embedding(pitch_tokens)
        else:
            pitch_tokens = torch.bucketize(pitch, self.pitch_bins)
            pitch_embedding = self.pitch_embedding(pitch_tokens)

        x = x + pitch_embedding



        prior_out = {
            "dur_pred_round": dur_pred_out["dur_pred_round"],
            "dur_pred_log": dur_pred_out["dur_pred_log"],
            "dur_pred": dur_pred_out["dur_pred"],
            "pitch_pred_log": pitch_pred_log,
            "pitch_token": pitch_tokens,
            "mel_len": mel_len,
            "prior_out": x,
        }

        return prior_out
