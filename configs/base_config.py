base_config = {
    "eval_interval": 1,
    "ema_rate": 0.999,
    "csv_keys": [ "Prompt", "IoU", "Recall", "Precision", "F1", "epoch"],
    "opt": {
        "learning_rate": 1e-5,
        "weight_decay": 1e-4,# L2 정규화를 위한 가중치 감쇠(regularization) 계수
        "decay_factor": 10,
        "steps": [3000, 8000],
        "warmup_steps": 250,
    },
    "corruptions": [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "snow",
        "frost",
        "fog",
        "brightness",
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
    ],
    "model": {
        "type": "vit_b",
        "checkpoint": "./pretrain/",
        "ckpt": "",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": True,
        },
    },
    "datasets": {
        "NWPU": {
            "root_dir": "/data2/hai_mw/PointSAM/data/NWPU/Images/positive image set",   
            "annotation_file_train": "/data2/hai_mw/PointSAM/data/NWPU/Annotations/NWPU_instances_train.json",
            "annotation_file_val": "/data2/hai_mw/PointSAM/data/NWPU/Annotations/NWPU_instances_val.json",
        },
        "WHU": {
            "root_dir": "/data2/hai_mw/PointSAM/data/WHU", 
            "annotation_file_train": "/data2/hai_mw/PointSAM/data/WHU/annotations/WHU_building_train.json",
            "annotation_file_val": "/data2/hai_mw/PointSAM/data/WHU/annotations/WHU_building_val.json",
        },
        "HRSID": {
            "root_dir": "/data2/hai_mw/PointSAM/data/HRSID/Images/images",   
            "annotation_file_train": "/data2/hai_mw/PointSAM/data/HRSID/Annotations/inshore/inshore_train.json",
            "annotation_file_val": "/data2/hai_mw/PointSAM/data/HRSID/Annotations/inshore/inshore_test.json"
        },
    },
}
