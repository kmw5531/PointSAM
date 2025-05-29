import os
import time
import argparse
import random
from abc import ABC

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import lightning as L
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.fabric import _FabricOptimizer

from box import Box
from datasets import call_load_dataset
from utils.model import Model
from utils.losses import DiceLoss, FocalLoss, Matching_Loss
from utils.eval_utils import AverageMeter, validate, get_prompts, calc_iou
from utils.tools import copy_model, create_csv, reduce_instances
from utils.utils import *
from utils.finch import FINCH

vis = False

def train_sam(
    cfg: Box,
    fabric: L.Fabric,
    model: Model,
    optimizer: _FabricOptimizer,
    scheduler: _FabricOptimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    target_pts,
):
    # 손실 함수 및 초기 변수 설정
    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    max_iou = 0.
    mem_bank = Store(1, cfg.mem_bank_max_len) 
    match_interval = cfg.match_interval
    # 에포크 및 배치 루프
    for epoch in range(1, cfg.num_epochs + 1):
        # 배치별 시간, 손실 측정을 위한 AverageMeter 객체 생성
        batch_time = AverageMeter()
        data_time = AverageMeter()
        focal_losses = AverageMeter()
        dice_losses = AverageMeter()
        iou_losses = AverageMeter()
        total_losses = AverageMeter()
        match_losses = AverageMeter()
        end = time.time()
        num_iter = len(train_dataloader)

        for iter, data in enumerate(train_dataloader):

            data_time.update(time.time() - end)
            images_weak, images_strong, bboxes, gt_masks, img_paths= data
            del data  
            
            batch_size = images_weak.size(0)
            num_insts = sum(len(gt_mask) for gt_mask in gt_masks)
            if num_insts > cfg.max_nums:
                bboxes, gt_masks = reduce_instances(bboxes, gt_masks, cfg.max_nums)
            prompts = get_prompts(cfg, bboxes, gt_masks)

            #1. caculate pairwise IoUs of masks
            mask_ious, init_masks = cal_mask_ious(cfg, model, images_weak, prompts, gt_masks)
       
            #2. 부정 프롬프트 보정을 통해 새로운 프롬프트 생성
            new_prompts = neg_prompt_calibration(cfg, mask_ious, prompts)

            #3. start training using new prompt
            soft_image_embeds, soft_masks, _, _ = model(images_weak, new_prompts)    # teacher : 부정 프롬프트 보정이 된 새로운 프롬프트를 통해 예측  
            
            if isinstance(soft_image_embeds, dict):
                soft_image_embeds = soft_image_embeds['vision_features']  
                
            _, pred_masks, iou_predictions, _= model(images_strong, prompts)   # student : 원본 프롬프트를 이용한 예측

            del _

            num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
            loss_focal = torch.tensor(0., device=fabric.device)
            loss_dice = torch.tensor(0., device=fabric.device)
            loss_match = torch.tensor(0., device=fabric.device)
            loss_iou = torch.tensor(0., device=fabric.device)
            
            for i, (embed, pred_mask, soft_mask,  gt_mask, prompt, iou_prediction) in enumerate(zip(soft_image_embeds, pred_masks, soft_masks, gt_masks, prompts, iou_predictions)):
                    
                soft_mask = (soft_mask > 0.).float()
                #  메모리 뱅크를 통한 FINCH 클러스터링 및 Matching Loss 계산
                # 메모리 뱅크 : 예측 피쳐를 저장 후, 충분한 데이터가 쌓이면 FINCH 클러스터링을 적용하여 Matching Loss를 계산합니다
                pred_feats = generate_predict_feats(cfg, embed, soft_mask, prompt)
                target_pts_ = target_pts['target_pts']  
                pred_feats = pred_feats.cpu().tolist()
                for pred_feat in pred_feats:
                    mem_bank.add([[pred_feat]], [0])
                if len(mem_bank.retrieve(0)) >= cfg.mem_bank_max_len and (iter + 1) % match_interval == 0:
                    pred_feats = mem_bank.retrieve(0)  
                    pred_feats = np.array(pred_feats) 

                    #FINCH
                    fin = FINCH(verbose=False)
                    results = fin.fit(pred_feats)
                    last_key = list(results.partitions.keys())[-1]
                    pred_pts = results.partitions[last_key]['cluster_centers']
                    loss_match += Matching_Loss(pred_pts, target_pts_, device = fabric.device)

                del embed
                # vis가 활성화된 경우, 이미지 및 마스크 결과를 저장하여 시각적으로 확인할 수 있습니다
                if vis:
                    img_name = os.path.basename(img_paths[i]).split('.')[0]
                    
                    image_weak = images_weak[0].permute(1,2,0).cpu().numpy()* 255
                    image_weak = cv2.cvtColor(image_weak, cv2.COLOR_BGR2RGB)

                    if vis:
                        for j in range(len(soft_mask)):
                            mask_iou = torch.max(mask_ious[j])
                            image_weak_ = image_weak.copy()
                            mask_area = torch.sum(gt_mask[j])
                            
                            gt_mask_np = gt_mask[j].cpu().numpy() * 255 
                            gt_mask_img = cv2.cvtColor(gt_mask_np, cv2.COLOR_GRAY2RGB)
                            
                            init_prompt_po = prompts[0][0][j][:cfg.num_points]
                            init_prompt_ne = prompts[0][0][j][cfg.num_points:]
                            
                            for po in init_prompt_po:
                                cv2.circle(image_weak_, (int(po[0]), int(po[1])), 12, (0, 0, 255), -1)
                                
                            init_mask_img = init_masks[j].cpu().detach().numpy() * 255
                            init_mask_img = cv2.cvtColor(init_mask_img, cv2.COLOR_GRAY2RGB)
                            for po,ne in zip(init_prompt_po,init_prompt_ne):
                                cv2.circle(init_mask_img, (int(po[0]), int(po[1])), 12, (0, 0, 255), -1)
                                cv2.circle(init_mask_img, (int(ne[0]), int(ne[1])), 12, (0, 255, 0), -1)
                                
                            prompt_po = new_prompts[0][0][j][:cfg.num_points]
                            prompt_ne = new_prompts[0][0][j][cfg.num_points:]
                            soft_mask_img = soft_mask[j].cpu().detach().numpy() * 255  

                            soft_mask_img = cv2.cvtColor(soft_mask_img, cv2.COLOR_GRAY2RGB)

                            for po,ne in zip(prompt_po,prompt_ne):
                                cv2.circle(soft_mask_img, (int(po[0]), int(po[1])), 12, (0, 0, 255), -1)
                                cv2.circle(soft_mask_img, (int(ne[0]), int(ne[1])), 12, (0, 255, 0), -1)

                            output_dir = "./save_mask_{}/{}/{}/".format(str(float(cfg.num_points)),cfg.dataset,str(epoch))
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            merged_image = concatenate_images_with_padding([image_weak_, gt_mask_img, init_mask_img, soft_mask_img])
                            img_name_ = '{}_{}_iou{}.jpg'.format(img_name,str(j),str(mask_iou))
                            if mask_iou>float(cfg.iou_thr) and mask_area>3000:
                                cv2.imwrite(os.path.join(output_dir,img_name_), merged_image) 
                del init_masks, mask_ious

                loss_focal += focal_loss(pred_mask, soft_mask, num_masks)
                loss_dice += dice_loss(pred_mask, soft_mask, num_masks)
                batch_iou = calc_iou(pred_mask, soft_mask)
                loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks
                
            del soft_image_embeds, pred_masks, iou_predictions, gt_masks 
            # 전체 손실 계산 및 역전파    
            loss_total = 20. * loss_focal + loss_dice + loss_iou + 0.1*loss_match            

            fabric.backward(loss_total)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()

            batch_time.update(time.time() - end)
            end = time.time()

            focal_losses.update(loss_focal.item(), batch_size)
            dice_losses.update(loss_dice.item(), batch_size)
            iou_losses.update(loss_iou.item(), batch_size)
            total_losses.update(loss_total.item(), batch_size)
            match_losses.update(loss_match.item(), batch_size)

            if (iter+1) %match_interval==0:
                # 에포크마다 validate 함수를 호출하여 검증하고, 최고 IoU를 기록하면 모델 체크포인트를 저장합니다.
                fabric.print(f'Epoch: [{epoch}][{iter + 1}/{len(train_dataloader)}]'
                             f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
                             f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'
                             f' | Focal Loss [{focal_losses.val:.4f} ({focal_losses.avg:.4f})]'
                             f' | Dice Loss [{dice_losses.val:.4f} ({dice_losses.avg:.4f})]'
                             f' | IoU Loss [{iou_losses.val:.4f} ({iou_losses.avg:.4f})]'
                             f' | Match Loss [{match_losses.val:.4f} ({match_losses.avg:.4f})]'
                             f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]')

            # loss_logger = {
            #     "Focal Loss": focal_losses.avg,
            #     "Dice Loss": dice_losses.avg,
            #     "IoU Loss": iou_losses.avg,
            #     "Total Loss": total_losses.avg
            # }
            # fabric.log_dict(loss_logger, num_iter * (epoch - 1) + iter)
            torch.cuda.empty_cache()
            
        if epoch % cfg.eval_interval == 0:
            iou, _= validate(fabric, cfg, model, val_dataloader, cfg.name, epoch)
            if iou > max_iou:
                state = {"model": model, "optimizer": optimizer}
                fabric.save(os.path.join(cfg.out_dir, "save", "best-ckpt.pth"), state)
                max_iou = iou
            del iou 
            
def configure_opt(cfg: Box, model: Model):

    def lr_lambda(step):
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        elif step < cfg.opt.steps[0]:
            return 1.0
        elif step < cfg.opt.steps[1]:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor**2)

    optimizer = torch.optim.Adam(model.model.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def corrupt_main(cfg):
    for corrupt in cfg.corruptions:
        cfg.corrupt = corrupt
        cfg.out_name = corrupt
        torch.cuda.empty_cache()
        main(cfg)


def main(cfg: Box) -> None:
    gpu_ids = [str(i) for i in range(torch.cuda.device_count())]
    num_devices = len(gpu_ids)
    fabric = L.Fabric(accelerator="auto",
                      devices=num_devices,
                      strategy="auto",
                      loggers=[TensorBoardLogger(cfg.out_dir)])
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)
    # 체크포인트 폴더 생성 및 CSV 파일 초기화: 학습 결과를 저장할 디렉토리를 만들고, 로그를 기록할 CSV 파일을 생성합니다.
    if fabric.global_rank == 0:
        os.makedirs(os.path.join(cfg.out_dir, "save"), exist_ok=True)
        create_csv(os.path.join(cfg.out_dir, "metrics.csv"), csv_head=cfg.csv_keys)
    # 모델 및 데이터셋 로드
    with fabric.device:
        model = Model(cfg)
        model.setup()

    load_datasets = call_load_dataset(cfg)
    train_data, val_data, pt_data = load_datasets(cfg, img_size=1024, return_pt = True)
    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)
    pt_data = fabric._setup_dataloader(pt_data)

    # 옵티마이저, 스케줄러 및 모델 체크포인트 로드
    optimizer, scheduler = configure_opt(cfg, model)
    model, optimizer = fabric.setup(model, optimizer)

    if cfg.resume and cfg.model.ckpt is not None:
        full_checkpoint = fabric.load(cfg.model.ckpt)
        model.load_state_dict(full_checkpoint["model"])
        optimizer.load_state_dict(full_checkpoint["optimizer"])
    print('-'*100)
    print('\033[92mDirect test on the original SAM.\033[0m') 
    _, _, = validate(fabric, cfg, model, val_data, name=cfg.name, epoch=0)
    print('-'*100)
    del _     
    
    # offline_prototypes_generation : gt값을 이용하여 프로토타입을 생성 
    target_pts = offline_prototypes_generation(cfg, model, pt_data)
    
    train_sam(cfg, fabric, model, optimizer, scheduler, train_data, val_data, target_pts)

    del model, train_data, val_data


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--cfg', help='train config file path')
    parser.add_argument('--prompt', help='the type of prompt')
    parser.add_argument('--num_points',type=int, help='the number of points')
    parser.add_argument('--out_dir', help='the dir to save logs and models')
    parser.add_argument('--load_type', help='the dir to save logs and models')      
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    print(torch.cuda.current_device())
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')
    args = parse_args()

    exec(f'from {args.cfg} import cfg')

    # transfer the args to a dict
    args_dict = vars(args)  
    cfg.merge_update(args_dict)

    main(cfg)
    torch.cuda.empty_cache()
