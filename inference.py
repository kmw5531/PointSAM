import os
import cv2
import copy
import torch
import argparse
import torch.nn.functional as F
import numpy as np
import segmentation_models_pytorch as smp
import lightning as L
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.fabric import _FabricOptimizer
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from box import Box
from datasets import call_load_dataset
from utils.model import Model
from utils.tools import copy_model, create_csv, reduce_instances
from utils.eval_utils import AverageMeter
from utils.sample_utils import uniform_sampling
from tqdm import tqdm 


torch.set_float32_matmul_precision('high')
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# 입력된 마스크에 대해 , 마스크 내의 객체의 중심 좌표를 계산하여 리스트로 반환 
def compute_centroids(masks):
    centroids = []
    for mask in masks:
        if not isinstance(mask, np.ndarray):
            # mask가 넘파이 배열이 아니면 cpu 텐서에서 넘파이 배열로 변환 
            mask = mask.cpu().numpy().astype(np.uint8)  
        
        # Find connected components
        # cv2.connectedComponentsWithStats 함수는 OpenCV에서 이진화된 이미지(보통 0과 1 또는 0과 255로 구성된 이미지)를 입력 받아,
        #  그 이미지 내에서 연결된 컴포넌트(connected components)를 식별하고, 각 컴포넌트에 대한 통계 정보를 계산해주는 함수
        # cv2.connectedComponentsWithStats를 사용하여 배경을 제외한 각 영역의 중심 좌표를 구함.
        num_labels, _, _, centroids_data = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # Extract centroids (skip background label 0)
        component_centroids = centroids_data[1:]  # Skip the first entry, which is for the background
        
        # Append centroids as a list
        centroids.append(component_centroids.tolist())
    #마스크 내의 객체의 중심 좌표를 계산하여 리스트로 반환 
    return centroids

def plotwithpoint(fabric: L.Fabric, anchor_model: Model, model: Model, val_dataloader: DataLoader):
    model.eval()
    anchor_model.eval()
    ious = AverageMeter() # iou 계산을 위한 객체
    f1_scores = AverageMeter()# f1_score 계산을 위한 객체  
    num_points = cfg.num_points
    # 전처리 함수 
    transform = val_dataloader.dataset.transform

    # 결과 저장 경로 
    save_path = cfg.out_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with torch.no_grad():
        for iter, data in enumerate(tqdm(val_dataloader, desc="Processing", unit="batch")):
            image_ids, paddings, ori_images, ori_bboxes, origin_masks, images, bboxes, gt_masks = data
            images = torch.stack(images).to(device=fabric.device)
            num_images = images.size(0)

            # gt 마스크로부터 양성 포인트와 음성 포인트를 추출하여 프롬프트로 만든다.   
            prompts = []
            for mask in gt_masks:
                po_points = compute_centroids(mask)
                na_points = uniform_sampling((~mask.to(bool)).to(float), num_points)
                po_point_coords = torch.tensor(po_points, device=fabric.device)
                na_point_coords = torch.tensor(na_points, device=fabric.device)
                # torch.cat을 사용하여, 양성과 음성 포인트 좌표를 하나의 텐서로 결합합니다.
                point_coords = torch.cat((po_point_coords, na_point_coords), dim=1)
                po_point_labels = torch.ones(po_point_coords.shape[:2], dtype=torch.int, device=fabric.device)
                na_point_labels = torch.zeros(na_point_coords.shape[:2], dtype=torch.int, device=fabric.device)
                # 양성은 1, 음성은 0으로 라벨링하고, 두 라벨 텐서를 결합합니다.
                point_labels = torch.cat((po_point_labels, na_point_labels), dim=1)
                in_points = (point_coords, point_labels)
                # (point_coords, point_labels) 형태의 튜플로 구성하여 prompts 리스트에 추가합니다.
                prompts.append(in_points)

            # 모델 추론 
            #최종적으로 네 가지 값을 반환합니다:
            #image_embeddings: 이미지 인코더의 출력, 주로 내부 특징 표현입니다.
            #pred_masks: 프롬프트를 기반으로 생성된 예측 마스크들.
            #ious: 예측된 IoU 값들(각 마스크에 대한 신뢰도 또는 품질 지표).
            #res_masks: 추가적인 마스크 출력(예: 다중 마스크 중 선택하지 않은 결과 등).
            
            # 교사모델과 학생 모델의 출력 중 예측 마스크만을 저장
            _, base_masks, _, _ = anchor_model(images, prompts)
            _, pred_masks, _, _ = model(images, prompts)

            draw_points = []
            for ori_mask in origin_masks:
                ori_po_points = uniform_sampling(ori_mask, num_points)
                draw_points.append(ori_po_points)

            for pred_mask, gt_mask in zip(pred_masks, gt_masks):
                batch_stats = smp.metrics.get_stats(
                    pred_mask,
                    gt_mask.to(device=fabric.device).int(),
                    mode='binary',
                    threshold=0.5,
                )
                batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                ious.update(batch_iou, num_images)
                f1_scores.update(batch_f1, num_images)
            fabric.print(
                f'Val:[{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
            )
            torch.cuda.empty_cache()

            for image_id, padding, base_mask, pred_mask, ori_mask, points, image in zip(image_ids, paddings, base_masks, pred_masks, origin_masks, draw_points, ori_images):
                H, W, C = image.shape

                base_mask = base_mask.unsqueeze(1)
                base_mask = base_mask[..., padding[1] : base_mask.shape[-2] - padding[3], padding[0] : base_mask.shape[-1] - padding[2]]
                base_mask = F.interpolate(base_mask, (H, W), mode="bilinear", align_corners=False)

                pred_mask = pred_mask.unsqueeze(1)
                pred_mask = pred_mask[..., padding[1] : pred_mask.shape[-2] - padding[3], padding[0] : pred_mask.shape[-1] - padding[2]]
                pred_mask = F.interpolate(pred_mask, (H, W), mode="bilinear", align_corners=False)

                fig, axs = plt.subplots(1, 4)
                fig.set_size_inches(W/100.0*4, H/100.0)

                image_0 = copy.deepcopy(image)
                image_1 = copy.deepcopy(image)
                image_2 = copy.deepcopy(image)
                image_3 = copy.deepcopy(image)
                axs[0].imshow(image_0)
                axs[1].imshow(image_1)
                axs[2].imshow(image_2)
                axs[3].imshow(image_3)
                axs[0].axis('off')
                axs[1].axis('off')
                axs[2].axis('off')
                axs[3].axis('off')

                masked_image_1 = np.zeros((H, W, 4))
                masked_image_2 = np.zeros((H, W, 4))
                masked_image_3 = np.zeros((H, W, 4))
                for point, ori_mask_i, base_mask_i, pred_mask_i in zip(points, ori_mask, base_mask, pred_mask):
                    color = np.random.random(3)
                    x_coords = []
                    y_coords = []
                    for point_i in point:
                        x, y = point_i
                        x_coords.append(x)
                        y_coords.append(y)
                    point_color = np.concatenate([color, [1.0]])
                    axs[0].scatter(x_coords, y_coords, c=point_color)

                    base_mask_i = (base_mask_i.squeeze(0) > 0.).cpu().numpy().astype(bool)
                    pred_mask_i = (pred_mask_i.squeeze(0) > 0.).cpu().numpy().astype(bool)
                    ori_mask_i = ori_mask_i.astype(bool)
                    mask_color = np.concatenate([color, [0.7]])

                    masked_image_1[ori_mask_i] = mask_color
                    axs[1].imshow(masked_image_1)
                    masked_image_2[base_mask_i] = mask_color
                    axs[2].imshow(masked_image_2)
                    masked_image_3[pred_mask_i] = mask_color
                    axs[3].imshow(masked_image_3)

                plt.subplots_adjust(wspace=0)
                plt.savefig(os.path.join(save_path, f"{image_id}.jpg"), dpi=100, bbox_inches='tight', pad_inches=0)
                plt.close(fig)

def main(cfg: Box, args) -> None:
    gpu_ids = [str(i) for i in range(torch.cuda.device_count())]
    num_devices = len(gpu_ids)

    fabric = L.Fabric(accelerator="auto",
                      devices=num_devices,
                      strategy="auto",
                      loggers=[TensorBoardLogger(cfg.out_dir)])
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    with fabric.device:
        anchor_model = Model(cfg)
        anchor_model.setup()

        model = Model(cfg)
        model.setup()
        full_checkpoint = fabric.load(args.ckpt)
        model.load_state_dict(full_checkpoint["model"])

    load_datasets = call_load_dataset(cfg)
    val_data = load_datasets(cfg, model.model.image_encoder.img_size)
    val_data = fabric._setup_dataloader(val_data)
    print(f"▶ Dataset에 등록된 이미지 수: {len(val_data.dataset)}")
    print(f"▶ 배치 사이즈: {val_data.batch_size}")
    print(f"▶ 총 배치(Iteration) 수: {len(val_data)}")

    plotwithpoint(fabric, anchor_model, model, val_data)

def parse_args():

    parser = argparse.ArgumentParser(description='Test a detector with specified config and checkpoint.')
    parser.add_argument(
        '--cfg', 
        type=str, 
        default="configs.config_hrsid", 
        help='Path to the configuration file (e.g., "configs.config_hrsid").'
    )
    parser.add_argument(
        '--out_dir', 
        type=str, 
        default="output", 
        help='Directory to save predicted results.'
    )
    parser.add_argument(
        '--ckpt', 
        type=str, 
        default="checkpoints/best-ckpt.pth", 
        help='Path to the model checkpoint file.'
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = parse_args()
    args_dict = vars(args)

    import importlib.util

    # --- 파일 경로에서 직접 cfg 모듈 로드 시작 ---
    cfg_path = args.cfg
    spec = importlib.util.spec_from_file_location("config", cfg_path)
    cfg_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_mod)
    cfg = cfg_mod.cfg
    # --- 파일 경로에서 직접 cfg 모듈 로드 끝 ---

    cfg.merge_update(args_dict)
    cfg.visual = True
    cfg.load_type = 'visual'
    
    main(cfg, args)