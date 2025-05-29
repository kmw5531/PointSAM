import os
import sys
import torch
import argparse
import torch.nn.functional as F
import numpy as np
import segmentation_models_pytorch as smp
import lightning as L
from lightning.fabric.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from box import Box
from datasets import call_load_dataset
from utils.model import Model
from utils.eval_utils import AverageMeter
from tqdm import tqdm
import types

# PointSAM/utils/sample_utils 경로 추가
base_dir = os.path.dirname(__file__)
utils_dir = os.path.abspath(os.path.join(base_dir, '..', 'PointSAM', 'utils'))
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)
from sample_utils import uniform_sampling

torch.set_float32_matmul_precision('high')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_pruned_state_dict(path):
    obj = torch.load(path, map_location='cpu')
    sd = obj.state_dict() if hasattr(obj, 'state_dict') else obj.get('model', obj)
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("image_encoder.module."):
            new_sd[k.replace("image_encoder.module.", "model.image_encoder.")] = v
        else:
            new_sd[k] = v
    return new_sd


def combine_qkv(pruned_sd):
    combined = {}
    for k in list(pruned_sd):
        if k.endswith(".attn.q.weight"):
            base = k[:-len(".attn.q.weight")]
            q = pruned_sd.pop(f"{base}.attn.q.weight")
            k_ = pruned_sd.pop(f"{base}.attn.k.weight")
            v = pruned_sd.pop(f"{base}.attn.v.weight")
            combined[f"{base}.attn.qkv.weight"] = torch.cat([q, k_, v], dim=0)
            qb = pruned_sd.pop(f"{base}.attn.q.bias")
            kb = pruned_sd.pop(f"{base}.attn.k.bias")
            vb = pruned_sd.pop(f"{base}.attn.v.bias")
            combined[f"{base}.attn.qkv.bias"] = torch.cat([qb, kb, vb], dim=0)
    pruned_sd.update(combined)
    return pruned_sd


def compute_centroids(masks):
    import cv2
    centroids = []
    for mask in masks:
        if not isinstance(mask, np.ndarray):
            mask = mask.cpu().numpy().astype(np.uint8)
        _, _, _, stats = cv2.connectedComponentsWithStats(mask, connectivity=8)
        centroids.append(stats[1:].tolist())
    return centroids


def evaluate_only(fabric: L.Fabric, model: Model, val_loader: DataLoader, cfg: Box):
    model.eval()
    ious = AverageMeter()
    f1s = AverageMeter()
    num_points = cfg.num_points

    with torch.no_grad():
        for it, data in enumerate(tqdm(val_loader, desc="Evaluating")):
            _, _, _, _, _, images, _, gt_masks = data
            images = torch.stack(images).to(fabric.device)

            # build prompts
            prompts = []
            for mask in gt_masks:
                po = compute_centroids(mask)
                na = uniform_sampling((~mask.to(bool)).to(float), num_points)
                po_c = torch.tensor(po, device=fabric.device)
                na_c = torch.tensor(na, device=fabric.device)
                coords = torch.cat((po_c, na_c), dim=1)
                labs = torch.cat([
                    torch.ones(po_c.shape[:2], dtype=torch.int, device=fabric.device),
                    torch.zeros(na_c.shape[:2], dtype=torch.int, device=fabric.device)
                ], dim=1)
                prompts.append((coords, labs))

            # inference
            _, pred_masks, _, _ = model(images, prompts)

            # per-image metrics
            for idx, (pm, gm) in enumerate(zip(pred_masks, gt_masks)):
                stats = smp.metrics.get_stats(pm, gm.to(fabric.device).int(),
                                              mode='binary', threshold=0.5)
                iou_i = smp.metrics.iou_score(*stats, reduction="micro-imagewise").item()
                f1_i = smp.metrics.f1_score(*stats, reduction="micro-imagewise").item()
                # 누적
                ious.update(iou_i, 1)
                f1s.update(f1_i, 1)
            fabric.print(
                f'Val:[{iter}/{len(val_loader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1s.avg:.4f}]'
            )
        # 최종 평균
        fabric.print(f'Final (avg) → IoU={ious.avg:.4f}, F1={f1s.avg:.4f}')


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate pruned checkpoint only.')
    p.add_argument('--cfg', required=True, help='config file')
    p.add_argument('--out_dir', default="output", help='save directory')
    p.add_argument('--prune_ckpt', required=True, help='pruned checkpoint')
    return p.parse_args()


def main(cfg: Box, args):
    raw = torch.load(args.prune_ckpt, map_location='cpu')
    sd = raw.state_dict() if hasattr(raw, 'state_dict') else raw.get('model', raw)

    # read embed_dim / patch_dim
    if 'image_encoder.module.pos_embed' in sd:
        pe = sd['image_encoder.module.pos_embed']
    elif 'model.image_encoder.pos_embed' in sd:
        pe = sd['model.image_encoder.pos_embed']
    else:
        raise KeyError("pos_embed key not found")

    if 'image_encoder.module.patch_embed.proj.weight' in sd:
        pw = sd['image_encoder.module.patch_embed.proj.weight']
    elif 'model.image_encoder.patch_embed.proj.weight' in sd:
        pw = sd['model.image_encoder.patch_embed.proj.weight']
    else:
        raise KeyError("patch_embed.proj.weight key not found")

    embed_dim, patch_dim = pe.shape[-1], pw.shape[0]
    if hasattr(cfg, 'model') and hasattr(cfg.model, 'image_encoder'):
        cfg.model.image_encoder.embed_dim = embed_dim
        cfg.model.image_encoder.patch_dim = patch_dim
        if hasattr(cfg.model.image_encoder, 'head_dim'):
            cfg.model.image_encoder.num_heads = embed_dim // cfg.model.image_encoder.head_dim
    else:
        cfg.embed_dim = embed_dim
        cfg.patch_dim = patch_dim

    fabric = L.Fabric(
        accelerator="auto",
        devices=list(range(torch.cuda.device_count())),
        strategy="auto",
        loggers=[TensorBoardLogger(cfg.out_dir)]
    )
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    with fabric.device:
        model = Model(cfg)
        model.setup()
        model = model.to(fabric.device)

        pruned_sd = load_pruned_state_dict(args.prune_ckpt)
        pruned_sd = combine_qkv(pruned_sd)
        missing, unexpected = model.load_state_dict(pruned_sd, strict=False)
        fabric.print(f">> Loaded pruned: missing={len(missing)} unexpected={len(unexpected)}")

        # patch forward
        def patched_forward(self, x):
            x = self.patch_embed(x)
            if self.pos_embed is not None:
                x = x + self.pos_embed
            for blk in self.blocks:
                x, *_ = blk(x)
            x = self.neck(x.permute(0, 3, 1, 2))
            return x

        enc = model.model.image_encoder
        if isinstance(enc, torch.nn.DataParallel):
            enc.module.forward = types.MethodType(patched_forward, enc.module)
        else:
            enc.forward = types.MethodType(patched_forward, enc)

    # dataset & loader
    ds_fn = call_load_dataset(cfg)
    if hasattr(model.model.image_encoder, 'img_size'):
        img_size = model.model.image_encoder.img_size
    else:
        img_size = model.model.image_encoder.module.img_size

    val_ds = ds_fn(cfg, img_size)
    val_loader = fabric._setup_dataloader(val_ds)
    print(f"▶ Samples: {len(val_loader.dataset)}, Batch size: {val_loader.batch_size}, Iter: {len(val_loader)}")

    # run eval only
    evaluate_only(fabric, model, val_loader, cfg)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location("config", args.cfg)
    cfg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_mod)
    cfg = cfg_mod.cfg

    cfg.merge_update(vars(args))
    cfg.out_dir = args.out_dir
    cfg.visual = True
    cfg.load_type = 'visual'

    main(cfg, args)
