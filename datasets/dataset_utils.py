import os
import copy
import random
import numpy as np
import random
import torch

datasets = {}
def register_dataset(name):
   def decorator(cls):
       datasets[name] = cls
       return cls
   return decorator

def make_dataset(name, is_training, split, **kwargs):
   dataset = datasets[name](is_training, split, **kwargs)
   return dataset

def make_data_loader(dataset, is_training, generator, sampler, batch_size, num_workers):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
        worker_init_fn=(worker_init_reset_seed if is_training else None),
        shuffle=False,
        drop_last=is_training,
        generator=generator,
        persistent_workers=True,
        sampler=sampler
    )
    return loader

def trivial_batch_collator(batch):
    return batch

def worker_init_reset_seed(worker_id):
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def trans_feats(
    data_dict,
    max_seq_len,
    trunc_thresh,
    crop_ratio=None,
    max_num_trials=200,
    has_action=True,
    no_trunc=False
):
    feat_len = data_dict['feats'].shape[1]
    num_segs = data_dict['segments'].shape[0]

    if feat_len <= max_seq_len:
        if crop_ratio == None:
            return data_dict
        else:
            max_seq_len = random.randint(
                max(round(crop_ratio[0] * feat_len), 1),
                min(round(crop_ratio[1] * feat_len), feat_len),
            )
            if feat_len == max_seq_len:
                return data_dict

    data_dict = copy.deepcopy(data_dict)

    for _ in range(max_num_trials):

        st = random.randint(0, feat_len - max_seq_len)
        ed = st + max_seq_len
        window = torch.as_tensor([st, ed], dtype=torch.float32)

        window = window[None].repeat(num_segs, 1)
        left = torch.maximum(window[:, 0], data_dict['segments'][:, 0])
        right = torch.minimum(window[:, 1], data_dict['segments'][:, 1])
        inter = (right - left).clamp(min=0)
        area_segs = torch.abs(
            data_dict['segments'][:, 1] - data_dict['segments'][:, 0])
        inter_ratio = inter / area_segs

        seg_idx = (inter_ratio >= trunc_thresh)

        if no_trunc:
            seg_trunc_idx = (inter_ratio > 0.0) & (inter_ratio < 1.0)
            if (seg_idx.sum().item() > 0) and (seg_trunc_idx.sum().item() == 0):
                break
        elif has_action:
            if seg_idx.sum().item() > 0:
                break
        else:
            break

    data_dict['feats'] = data_dict['feats'][:, st:ed].clone()
    data_dict['segments'] = torch.stack((left[seg_idx], right[seg_idx]), dim=1)
    data_dict['segments'] = data_dict['segments'] - st
    data_dict['labels'] = data_dict['labels'][seg_idx].clone()

    return data_dict
