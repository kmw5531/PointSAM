# configs/config_whu_rsi.py
from box import Box
from configs.base_config import base_config

# 1) 먼저 base_config 를 로드
cfg = Box(base_config)

# 2) WHU 전용 설정
cfg.merge_update({
    "dataset": "WHU",
    # inference 단계에서 visual=True 로 덮어씌워 주므로 여기서는 soft 로 설정
    "load_type": "soft",
    "num_points": 3,

    # DataLoader 관련
    "batch_size": 1,
    "val_batchsize": 1,
    "num_workers": 0,
    # PointSAM이 참조하는 prompt 모드
    "prompt": "point",
    # pruning 된 모델의 image_encoder 설정
    "model": {
        "type": "vit_b",
        # ==== prune step2 에서 뽑아낸 값들 ====
        # pos_embed.shape[-1] == 384
        # patch_embed.proj.weight.shape[0] == 384
        # q/k/v 가 6 heads 로 나뉘어지고 (384/64==6)
        # mlp hidden dim == 384 * 4 == 1536
        "image_encoder": {
            "embed_dim": 384,
            "patch_dim": 384,
            "num_heads": 6,
            "head_dim": 64,   # embed_dim / num_heads
            "mlp_ratio": 4,   # hidden_dim = embed_dim * mlp_ratio = 1536
        },
        # neck 의 출력 채널은 checkpoint 에서 (256,384,1,1) 이었으니 그대로 256 사용
        "neck": {
            "out_channels": 256
        }
    },

    # LoRA 관련 설정 (inference 시 LoRA off 또는 필요에 따라 rank 조정)
    "start_lora_layer": 1,
    "lora_rank": 0,       # inference 단계에서는 LoRA 비활성화를 위해 0

    # inference 에선 아래 두 필드를 override 합니다
    "visual": False,
    "out_dir": "",

    # optimizer 설정 (inference 에서는 사용 안 함)
    "opt": {
        "learning_rate": 5e-4,
        "weight_decay": 1e-4
    }
})

# 이 파일을 import 해서 cfg 변수를 사용하세요.
