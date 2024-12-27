# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import torch
from torch.nn.utils.rnn import pad_sequence
from utils.data_utils import *
from processors.acoustic_extractor import cal_normalized_mel
from processors.acoustic_extractor import load_normalized
from models.base.base_dataset import (
    BaseOfflineCollator,
    BaseOfflineDataset,
    BaseTestDataset,
    BaseTestCollator,
)
from new_text.cleaner import clean_text
from new_text import cleaned_text_to_sequence
from tqdm import tqdm
import pickle
import tgt

class NS2Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg, dataset, is_valid=False):
        assert isinstance(dataset, str)

        processed_data_dir = os.path.join(cfg.preprocess.processed_dir, dataset)
        self.processed_data_dir = processed_data_dir

        meta_file = cfg.preprocess.valid_file if is_valid else cfg.preprocess.train_file
        # train.json

        self.metafile_path = os.path.join(processed_data_dir, meta_file)

        self.metadata = self.get_metadata()

        self.sampling_rate=24000
        self.hop_length=320
        # get phone to id / id to phone map
        # self.phone2id, self.id2phone = self.get_phone_map()

        self.all_num_frames = []
        for i in range(len(self.metadata)):
            # self.all_num_frames.append(self.metadata[i]["num_frames"])
            self.all_num_frames.append(self.metadata[i]["Duration"]*75)
        self.num_frame_sorted = np.array(sorted(self.all_num_frames))
        self.num_frame_indices = np.array(
            sorted(
                range(len(self.all_num_frames)), key=lambda k: self.all_num_frames[k]
            )
        )
        
        self.cfg = cfg

        # self.all_codes = []
        # self.all_pitches = []
        # self.all_durations = []
        self.all_phone_ids = []
        self.all_tone_ids = []
        # self.all_frame_nums = []

        for utt_info in tqdm(self.metadata):
            # 加载代码
            # path_formatted_uid = utt_info["Uid"]
            text = utt_info["Text"]
            norm_text, phone, tone, word2ph = clean_text(text, 'ZH')
            phone, tone, language = cleaned_text_to_sequence(phone, tone, 'ZH')
 

            # 加载对齐信息并计算持续时间和音素ID
            # textgrid_path = os.path.join(cfg.dataset_path[utt_info["Dataset"]], 'PhoneLabeling', f"{path_formatted_uid}.interval")
            # textgrid = tgt.io.read_textgrid(textgrid_path)
            # phone, duration, _, _ = self.get_alignment(textgrid.tiers[0])
            # phone_id = np.array([self.phone2id.get(p) for p in phone])

            # self.all_durations.append(duration)
            self.all_phone_ids.append(phone)
            self.all_tone_ids.append(tone)

            # 计算帧数
            # frame_nums = code.shape[1]
            # self.all_frame_nums.append(frame_nums)



    def __len__(self):
        return len(self.metadata)

    def get_dataset_name(self):
        return self.metadata[0]["Dataset"]

    def get_metadata(self):
        with open(self.metafile_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        print("metadata len: ", len(metadata))

        return metadata

    # def get_phone_map(self):
    #     symbols = ["_"] + valid_symbols + ["sp", "sp1", "sil"] + ["<s>", "</s>"]+["<br>"]
    #     phone2id = {s: i for i, s in enumerate(symbols)}
    #     id2phone = {i: s for s, i in phone2id.items()}
    #     self.symbols = symbols
    #     return phone2id, id2phone
 
    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            if phones == []:
                if p in sil_phones:
 
                    phones.append("<s>")
                else:
 
                    phones.append("<s>")
                    durations.append(int(0))  
                    phones.append(p)
            else:
                if p not in sil_phones:
                    # For ordinary phones
                    phones.append(p)
                    # end_time = e
                    # end_idx = len(phones)
                else:
                    # For silent phones
                    phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        # phones = phones[:end_idx]
        # durations = durations[:end_idx]
        if phones[-1] in sil_phones:
            phones[-1] = "</s>"
        else:
            phones.append("</s>")
            durations.append(int(0))
        return phones, durations, start_time, end_time
         

    def __getitem__(self, index):
        utt_info = self.metadata[index]

        # dataset = utt_info["Dataset"]
        uid = utt_info["Uid"]
        # utt = "{}_{}".format(dataset, uid)

        single_feature = dict()


        path_formatted_uid = uid

        path_code = os.path.join(self.processed_data_dir, 'code',f"{path_formatted_uid}.npy")
 
        code = np.load(path_code).transpose()
        frame_nums = code.shape[1]
        # pitch
        pitch_root_path = os.path.join(self.processed_data_dir, 'pitch')
        pitch_path = os.path.join(pitch_root_path, f"{path_formatted_uid}.npy")
        pitch = np.load(pitch_path)
  
        # textgrid_path = os.path.join('/scratch/buildlam/speech_yz/LibriTTS_text_grid_11',f"{path_formatted_uid}.TextGrid")
        # textgrid = tgt.io.read_textgrid(textgrid_path)
        # phone, duration, _, _ = self.get_alignment(
        #     textgrid.get_tier_by_name("phones")
        # )
        # phone_id = np.array(
        #     [
        #         *map(
        #             self.phone2id.get,
        #             phone,
        #         )
        #     ]
        # )
 


        # code = self.all_codes[index]
        # pitch = self.all_pitches[index]
        # duration = self.all_durations[index]
        phone_id = self.all_phone_ids[index]
        tone_id = self.all_tone_ids[index]
        # frame_nums = self.all_frame_nums[index]

        # 获取说话人ID
        # spkid = self.metadata[index]["Singer"]
        code, pitch, phone_id, frame_nums = self.align_length(
            code, pitch, phone_id, frame_nums
        )

        # spkid
        # spkid = self.utt2spkid[utt]

        # get target and reference
        out = self.get_target_and_reference(code, pitch, phone_id, tone_id, frame_nums)
        code, ref_code = out["code"], out["ref_code"]
        pitch, ref_pitch = out["pitch"], out["ref_pitch"]
        phone_id = out["phone_id"]
        tone_id = out["tone_id"]
        frame_nums, ref_frame_nums = out["frame_nums"], out["ref_frame_nums"]

        single_feature.update(
            {
                "code": code,
                "frame_nums": frame_nums,
                "pitch": pitch,
                "phone_id": phone_id,
                "ref_code": ref_code,
                "ref_frame_nums": ref_frame_nums,
                "ref_pitch": ref_pitch,
                "tone_id": tone_id,
            }
        )

        return single_feature

    def get_num_frames(self, index):
        utt_info = self.metadata[index]
 
        return utt_info["Duration"]*75

    def align_length(self, code, pitch, phone_id, frame_nums):
        # aligh lenght of code, pitch, duration, phone_id, and frame nums
        code_len = code.shape[1]
        pitch_len = len(pitch)
        min_len = min(code_len, pitch_len)
        code = code[:, :min_len]
        # if pitch_len >= min_len:
        pitch = pitch[:min_len]
        # else:
        #     pitch = np.pad(pitch, (0, min_len - pitch_len), mode="edge")
        frame_nums = min_len
        # if dur_sum > min_len:
        # assert (duration[-1] - (dur_sum - min_len)) >= 0
        # duration[-1] = duration[-1] - (dur_sum - min_len)
        # assert duration[-1] >= 0

        return code, pitch, phone_id, frame_nums

    def get_target_and_reference(self, code, pitch, phone_id, tone_id, frame_nums):
        phone_nums = len(phone_id)
        clip_frame_nums = np.random.randint(
            int(frame_nums * 0.1), int(frame_nums * 0.5) + 1
        )
        clip_frame_nums = max(clip_frame_nums, 1)
        assert clip_frame_nums < frame_nums and clip_frame_nums >= 1
        if self.cfg.preprocess.clip_mode == "mid":
            start_frames = np.random.randint(0, frame_nums - clip_frame_nums)
        elif self.cfg.preprocess.clip_mode == "start":
            start_frames = 0
        else:
            assert self.cfg.preprocess.clip_mode in ["mid", "start"]
        end_frames = start_frames + clip_frame_nums

        new_code = np.concatenate(
            (code[:, :start_frames], code[:, end_frames:]), axis=1
        )
        ref_code = code[:, start_frames:end_frames]

        new_pitch = np.append(pitch[:start_frames], pitch[end_frames:])
        ref_pitch = pitch[start_frames:end_frames]


        new_frame_nums = frame_nums - (end_frames - start_frames)
        ref_frame_nums = end_frames - start_frames

        return {
            "code": code,
            "ref_code": ref_code,
            "pitch": pitch,
            "ref_pitch": ref_pitch,
            "phone_id": phone_id,
            "tone_id": tone_id,
            "frame_nums": frame_nums,
            "ref_frame_nums": ref_frame_nums,
        }


 


class NS2Collator(BaseOfflineCollator):
    def __init__(self, cfg):
        BaseOfflineCollator.__init__(self, cfg)

    def __call__(self, batch):
        packed_batch_features = dict()

        # code: (B, 16, T)
        # frame_nums: (B,)   not used
        # pitch: (B, T)
        # duration: (B, N)
        # phone_id: (B, N)
        # phone_id_frame: (B, T)
        # ref_code: (B, 16, T')
        # ref_frame_nums: (B,)   not used
        # ref_pitch: (B, T)   not used
        # ref_duration: (B, N')   not used
        # ref_phone_id: (B, N')   not used
        # ref_phone_frame: (B, T')   not used
        # spkid: (B,)   not used
        # phone_mask: (B, N)
        # mask: (B, T)
        # ref_mask: (B, T')

        for key in batch[0].keys():
            if key == "phone_id":
                phone_ids = [torch.LongTensor(b["phone_id"]) for b in batch]
                tone_ids = [torch.LongTensor(b["tone_id"]) for b in batch]
                packed_batch_features["phone_lengths"] = torch.LongTensor([len(b["phone_id"]) for b in batch])
                phone_masks = [torch.ones(len(b["phone_id"])) for b in batch]
                packed_batch_features["phone_id"] = pad_sequence(
                    phone_ids,
                    batch_first=True,
                    padding_value=0,
                )
                packed_batch_features["phone_mask"] = pad_sequence(
                    phone_masks,
                    batch_first=True,
                    padding_value=0,
                )
                packed_batch_features["tone_id"] = pad_sequence(
                    tone_ids,
                    batch_first=True,
                    padding_value=0,
                )
            elif key == "ref_code":
                ref_codes = [
                    torch.from_numpy(b["ref_code"]).transpose(0, 1) for b in batch
                ]
                ref_masks = [torch.ones(max(b["ref_code"].shape[1], 1)) for b in batch]
                packed_batch_features["ref_code"] = pad_sequence(
                    ref_codes,
                    batch_first=True,
                    padding_value=0,
                ).transpose(1, 2)
                packed_batch_features["ref_mask"] = pad_sequence(
                    ref_masks,
                    batch_first=True,
                    padding_value=0,
                )
            elif key == "code":
                codes = [torch.from_numpy(b["code"]).transpose(0, 1) for b in batch]
                masks = [torch.ones(max(b["code"].shape[1], 1)) for b in batch]
                packed_batch_features["code"] = pad_sequence(
                    codes,
                    batch_first=True,
                    padding_value=0,
                ).transpose(1, 2)
                packed_batch_features["mask"] = pad_sequence(
                    masks,
                    batch_first=True,
                    padding_value=0,
                )
            elif key == "pitch":
                values = [torch.from_numpy(b[key]) for b in batch]
                packed_batch_features[key] = pad_sequence(
                    values, batch_first=True, padding_value=50.0
                )
            elif key == "frame_nums":
                packed_batch_features["frame_nums"] = torch.LongTensor(
                    [b["frame_nums"] for b in batch]
                )
            elif key == "ref_frame_nums":
                packed_batch_features["ref_frame_nums"] = torch.LongTensor(
                    [b["ref_frame_nums"] for b in batch]
                )
            else:
                pass

        return packed_batch_features


def _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
    if len(batch) == 0:
        return 0
    if len(batch) == max_sentences:
        return 1
    if num_tokens > max_tokens:
        return 1
    return 0


def batch_by_size(
    indices,
    num_tokens_fn,
    max_tokens=None,
    max_sentences=None,
    required_batch_size_multiple=1,
):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        num_tokens_fn (callable): function that returns the number of tokens at
            a given index
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be a multiple of N (default: 1).
    """
    bsz_mult = required_batch_size_multiple

    sample_len = 0
    sample_lens = []
    batch = []
    batches = []
    for i in range(len(indices)):
        idx = indices[i]
        num_tokens = num_tokens_fn(idx)
        sample_lens.append(num_tokens)
        sample_len = max(sample_len, num_tokens)

        assert (
            sample_len <= max_tokens
        ), "sentence at index {} of size {} exceeds max_tokens " "limit of {}!".format(
            idx, sample_len, max_tokens
        )
        num_tokens = (len(batch) + 1) * sample_len

        if _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
            mod_len = max(
                bsz_mult * (len(batch) // bsz_mult),
                len(batch) % bsz_mult,
            )
            batches.append(batch[:mod_len])
            batch = batch[mod_len:]
            sample_lens = sample_lens[mod_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
        batch.append(idx)
    if len(batch) > 0:
        batches.append(batch)
    return batches
