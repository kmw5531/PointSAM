from box import Box
from configs.base_config import base_config

config = {
    "dataset": "HRSID", 
    "load_type": "soft",
    "num_points": 3,
    
    "batch_size": 1, #only support 1
    "val_batchsize": 1,
    "num_workers": 0,
    "num_epochs": 10,
    "max_nums": 50, # 한 이미지 내의 최대 인스턴스 수
    "resume": False,

    "start_lora_layer": 1,
    "lora_rank": 4,
    "mem_bank_max_len": 512, # 메모리 뱅크에 저장할 최대 특징 벡터 수 
    "match_interval": 30,# 매칭/클러스터링 검증(FINCH) 수행 주기
    "iou_thr": 0.1,

    "prompt": "point",
    "out_dir": "",
    "name": "base",
    "corrupt": None,
    "visual": False,
    "model": {
        "type": "vit_b", 
        },
    "opt": {
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [1500, 2000],
        "warmup_steps": 250,
    },
}

cfg = Box(base_config)
cfg.merge_update(config)
