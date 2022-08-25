import yaml

DEFAULTS = {
    "init_rand_seed": 99999999999,
    "dataset_name": "thumos",
    "devices": ['cuda:0'],
    "train_split": ('training', ),
    "val_split": ('validation', ),
    "model_name": "AdaPerFormer",
    "dataset": {
        "feat_stride": 16,
        "default_fps": None,
        "input_dim": 2304,
        "num_frames": 32,
        "num_classes": 100,
        "downsample_rate": 1,
        "max_seq_len": 2304,
        "trunc_thresh": 0.5,
        "crop_ratio": None,
        "force_upsampling": False,
    },
    "loader": {
        "batch_size": 8,
        "num_workers": 4,
    },
    "model": {
        "n_head": 4,
        "n_mha_win_size": -1,
        "embd_kernel_size": 3,
        "embd_dim": 512,
        "embd_use_ln": True,
        "head_dim": 512,
        "head_kernel_size": 3,
        "head_use_ln": True,
        "abs_pe": False,
        "rel_pe": False,
    },
    "train_cfg": {
        "loss_weight": 1.0, 
        "cls_prior_prob": 0.01,
        "init_loss_norm": 2000,
        "clip_grad_l2norm": -1,
        "head_empty_cls": [],
        "dropout": 0.1,
        "droppath": 0.1,
        "label_smoothing": 0.0,
        "center_sample": "radius",
        "center_sample_radius": 1.5,
    },
    "test_cfg": {
        "min_score": 0.01,
        "max_seg_num": 1000,
        "duration_thresh": 0.05,
        "multiclass_nms": True,
        "ext_score_file": None,
        "voting_thresh" : 0.75,
        "pre_nms_thresh": 0.001,
        "pre_nms_topk": 5000,
        "iou_threshold": 0.1,
    },

    "opt": {
        "weight_decay": 0.0,
        "learning_rate": 1e-3,
        "epochs": 30,
        "warmup": True,
        "warmup_epochs": 5,
        "schedule_type": "cosine",
        "schedule_steps": [],
        "schedule_gamma": 0.1,
        "type": "AdamW",
        "momentum": 0.9,
    }
}

def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v

def load_default_config():
    config = DEFAULTS
    return config

def _update_config(config):
    # fill in derived fields
    config["model"]["input_dim"] = config["dataset"]["input_dim"]
    config["model"]["num_classes"] = config["dataset"]["num_classes"]
    config["model"]["max_seq_len"] = config["dataset"]["max_seq_len"]
    config["model"]["train_cfg"] = config["train_cfg"]
    config["model"]["test_cfg"] = config["test_cfg"]
    return config

def load_config(config_file, defaults=DEFAULTS):
    use open(config_file, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    _merge(defaults, config)
    config = _update_config(config)
    return config
