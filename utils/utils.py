import random
from collections import deque
from tqdm import tqdm
from box import Box
import numpy as np
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
import torch.nn.functional as F

from .finch import FINCH
from .sample_utils import get_point_prompts

class Store:
    def __init__(self, total_num_classes, items_per_class, shuffle=False):
        self.shuffle = shuffle
        self.items_per_class = items_per_class
        self.total_num_classes = total_num_classes
        self.store = [deque(maxlen=self.items_per_class) for _ in range(self.total_num_classes)]

    def add(self, items, class_ids):
        for idx, class_id in enumerate(class_ids):
            self.store[class_id].append(items[idx])

    def retrieve(self, class_id):
        if class_id != -1:
            items = []
            for item in self.store[class_id]:
                items.extend(list(item))
            if self.shuffle:
                random.shuffle(items)
            return items
        else:
            all_items = []
            for i in range(self.total_num_classes):
                items = []
                for item in self.store[i]:
                    items.append(list(item))
                all_items.append(items)
            return all_items

    def reset(self):
        self.store = [deque(maxlen=self.items_per_class) for _ in range(self.total_num_classes)]

    def __str__(self):
        s = self.__class__.__name__ + '('
        for idx, item in enumerate(self.store):
            s += '\n Class ' + str(idx) + ' --> ' + str(len(list(item))) + ' items'
        s = s + ' )'
        return s

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return sum([len(s) for s in self.store])

def concatenate_images_with_padding(images, padding=10, color=(255, 255, 255)):
    heights = [image.shape[0] for image in images]
    widths = [image.shape[1] for image in images]

    total_width = sum(widths) + padding * (len(images) - 1)
    max_height = max(heights)
    
    if len(images[0].shape) == 3:
        new_image = np.full((max_height, total_width, 3), color, dtype=np.uint8)
    else:
        new_image = np.full((max_height, total_width), color[0], dtype=np.uint8)

    x_offset = 0
    for image in images:
        new_image[0:image.shape[0], x_offset:x_offset + image.shape[1]] = image
        x_offset += image.shape[1] + padding
    
    return new_image

def calculate_iou(mask1, mask2):
    intersection = torch.logical_and(mask1, mask2)
    union = torch.logical_or(mask1, mask2)
    iou = torch.sum(intersection).float() / torch.sum(union).float()
    return iou

def calc_iou_matrix(mask_list1, mask_list2):
    iou_matrix = torch.zeros((len(mask_list1), len(mask_list2)))
    for i, mask1 in enumerate(mask_list1):
        for j, mask2 in enumerate(mask_list2):
            iou_matrix[i, j] = calculate_iou(mask1, mask2)
    return iou_matrix

def cal_mask_ious(
    cfg,
    model,
    images_weak,
    prompts,
    gt_masks,
):
    with torch.no_grad():
         _, soft_masks, _, _ = model(images_weak, prompts)   

    for i, (soft_mask, gt_mask) in enumerate(zip(soft_masks, gt_masks)):  
        soft_mask = (soft_mask > 0).float()
        mask_ious = calc_iou_matrix(soft_mask, soft_mask)
        indices = torch.arange(mask_ious.size(0))
        mask_ious[indices, indices] = 0.0
    return mask_ious, soft_mask


def neg_prompt_calibration(
    cfg,
    mask_ious,
    prompts,
):
    '''
    mask_ious:[mask_nums,mask_nums]
    '''
    point_list = []
    point_labels_list = []
    num_points = cfg.num_points
    for m in range(len(mask_ious)):
            
        pos_point_coords = prompts[0][0][m][:num_points].unsqueeze(0) 
        neg_point = prompts[0][0][m][num_points:].unsqueeze(0)  
        neg_points_list = []
        neg_points_list.extend(neg_point[0])

        indices = torch.nonzero(mask_ious[m] > float(cfg.iou_thr)).squeeze(1)

        if indices.numel() != 0:
            # neg_points_list = []
            for indice in indices:
                neg_points_list.extend(prompts[0][0][indice][:num_points])
            neg_points = random.sample(neg_points_list, num_points)
        else:
            neg_points =neg_points_list
            
        neg_point_coords = torch.tensor([p.tolist() for p in neg_points], device=neg_point.device).unsqueeze(0)

        point_coords = torch.cat((pos_point_coords, neg_point_coords), dim=1) 

        point_list.append(point_coords)
        pos_point_labels = torch.ones(pos_point_coords.shape[0:2], dtype=torch.int, device=neg_point.device)
        neg_point_labels = torch.zeros(neg_point_coords.shape[0:2], dtype=torch.int, device=neg_point.device)
        point_labels = torch.cat((pos_point_labels, neg_point_labels), dim=1)  
        point_labels_list.append(point_labels)

    point_ = torch.cat(point_list).squeeze(1)
    point_labels_ = torch.cat(point_labels_list)
    new_prompts = [(point_, point_labels_)]
    return new_prompts
# 학습이나 추론에 사용될 프롬프트(결국은 get_point_prompts함수만 사용)를 생성하는 함수
def get_prompts(cfg: Box, bboxes, gt_masks):
    if cfg.prompt == "box" or cfg.prompt == "coarse":
        prompts = bboxes
    elif cfg.prompt == "point": # 이 논문에선 point 만을 사용함 
        # 양성(positive) 포인트와음성(negative) 포인트를 동일 개수(num_points)만큼 샘플링하여,
        # 각각의 인스턴스(마스크)에 대해 프롬프트 형태로 생성하고 (point_coords(좌표), point_labels) 형태로 반환.
        prompts = get_point_prompts(gt_masks, cfg.num_points)
    else:
        raise ValueError("Prompt Type Error!")
    return prompts # (point_coords, point_labels) 형태로 반환

def generate_predict_feats(cfg, embed, pseudo_label, gts):
    coords, lbls = gts
    selected_coords = []
    
    num_insts = len(pseudo_label)
    num_points = cfg.num_points
    for coord_grp, lbl_grp in zip(coords, lbls):
        for coord, lbl in zip(coord_grp, lbl_grp):  
            if lbl.item() == 1:  
                selected_coords.append(coord.tolist())

    # Downsample coordinates (SAM's stride is 16)
    coords = [[int(c // 16) for c in pair] for pair in selected_coords]

    embed = embed.permute(1, 2, 0)  # [H, W, C]

    pos_pts = [] 

    for index in range(0, num_insts * num_points, num_points):
        index = random.randint(0, num_points - 1)
        x, y = coords[index]
        pos_pt = embed[x, y]
        pos_pts.append(pos_pt)

    predict_feats = torch.stack(pos_pts, dim=0)

    return predict_feats

# 데이터로더에서 이미지를 받아 동결된 이미지 인코더를 통해 Feature embedding을 추출하고
# FINCH를 통해 클러스터링을 수행하여 프로토타입(대표 벡터)을 생성한 뒤 딕셔너리 형태로 반환
def offline_prototypes_generation(cfg, model, loader):
    model.eval()
    pts = [] # 임베딩 벡터를 저장할 리스트
    max_iters = 128 
    num_points = cfg.num_points

    #데이터 로더 반복
    with torch.no_grad():
        # tqdm은 반복문을 감싸서 진행 상황을 시각적으로 보여주는 라이브러리
        # desc는 진행 상황을 설명하는 문자열, ncols는 진행 표시줄의 너비를 설정
        # enumerate는 반복 가능한 객체(여기서는 batch)를 인덱스와 함께 반복하는 함수
        for i, batch in enumerate(tqdm(loader, desc='Generating target prototypes', ncols=100)):
            if i >= max_iters: # 128배치까지만 
                break
            #배치에서 이미지, 바운딩 박스, 마스크를 가져옴
            imgs, boxes, masks, _ = batch
            # (point_coords, point_labels) 형태로 점 프롬프트(gt 값)를 반환받음
            prompts = get_prompts(cfg, boxes, masks) 

            # 이미지를 모델에 넣고, prompts(양성·음성 점) 정보를 함께 주어 임베딩(혹은 마스크 예측) 등을 받음.
            # 즉 생성한 점 프롬프트를 동결된 sam모델에 넣어 embedding과 mask를 추출함
            embeds, masks, _, _ = model(imgs, prompts) 
            # _는 사용하지 않는 변수로 참조를 제거하여 메모리 해제
            del _

            # 모델의 출력에서 'vision_features' 키를 사용하여 임베딩을 가져옴 , 이 임베딩은 동결된 이미지 인코더의 Feature Map을 의미
            if isinstance(embeds, dict):
                embeds = embeds['vision_features'] 
            
            # 임베딩에서 양성 포인트 좌표 추출
            for embed, prompt, mask in zip(embeds, prompts, masks):
                # 이미지(또는 배치) 안에 존재하는 “인스턴스의 수”
                num_insts = len(mask)
                embed = embed.permute(1, 2, 0)  # [H, W, C]
                # 양성 포인트의 (x, y) 위치들을 저장할 공간
                coords = []

                points, labels = prompt
                for point_grp, label_grp in zip(points, labels):
                    for point, label in zip(point_grp, label_grp): 
                        if label.item() == 1:
                            # 양성 포인트일시 리스트에 추가   
                            coords.append(point.tolist())

                # 16 is the stride of SAM
                # SAM 인코더 특징맵의 해상도가 원본의 1/16로 축소 , 논문에서 말하는 (x/s , y/s)에서 s는 16을 의미
                coords = [[int(pt / 16) for pt in pair] for pair in coords]
                # 양성 포인트 개수만큼만 접근함 
                for index in range(0, num_insts*num_points, num_points): 
                    x, y = coords[index]
                    pt = embed[x,y] # 위에서 embed = embed.permute(1, 2, 0)을 해놓았으므로, embed[x, y]는 (x, y) 위치의 임베딩 벡터
                    #모든 양성 포인트 임베딩이 pts에 누적
                    pts.append(pt)
    # FinCH 클러스터링을 통해 프로토타입(대표 벡터)을 생성
    # FINCH는 클러스터링 알고리즘으로, 입력된 포인트들을 클러스터링하여 각 클러스터의 중심을 찾음
    # 이 중심점들이 프로토타입으로 사용됨
    fin = FINCH(verbose=True)
    pts = torch.stack(pts).cpu().numpy() # FINCH는 넘파이 기반으로 동작하므로 이 변환이 필요.
    # FINCH 알고리즘을 수행하여 pts를 군집화(클러스터링).
    # 결과 res에는 클러스터 파티션들이 여러 단계(partitions)로 저장됨.
    res = fin.fit(pts)
    # “가장 마지막 단계”에 해당하는 키(= last_key)를 선택하여 최종 군집 결과를 사용
    last_key = list(res.partitions.keys())[-1]
    # 해당 파티션의 군집(클러스터) 각 중심점(centroid) 벡터
    # 출력 형태 :
    #Clusters in 0 partition: 128
    #Clusters in 1 partition: 16
    #Clusters in 2 partition: 4
    #Clusters in 3 partition: 2 
    # => 마지막으로 2개의 클러스트로 나눠진 결과를 사용하는 것
    pt_stats = {'target_pts': res.partitions[last_key]['cluster_centers']}
    return pt_stats


