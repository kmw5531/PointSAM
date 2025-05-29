import os
import cv2
import torch
# import kornia as K
import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union
from imagecorruptions import corrupt, get_corruption_names

weak_transforms = A.Compose(
    [A.Flip()], # 이미지 좌우봔전만 수행행
    bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_ids"]),
    # keypoint_params=A.KeypointParams(format='xy')
)

strong_transforms = A.Compose(
#A.Posterize(): 이미지의 색상 수를 줄여서 픽셀 값을 제한합니다.
# A.Equalize(): 히스토그램 평활화를 통해 이미지의 명암을 균등하게 만듭니다.
# A.Sharpen(): 이미지의 선명도를 높여 경계를 강조합니다.
# A.Solarize(): 태양의 효과처럼 특정 밝기 이상의 영역을 반전시키는 효과.
# A.RandomBrightnessContrast(): 무작위로 밝기와 대비를 조절합니다.
# A.RandomShadow(): 이미지에 그림자 효과를 추가합니다.
    [
        A.Posterize(), 
        A.Equalize(),
        A.Sharpen(),
        A.Solarize(),
        A.RandomBrightnessContrast(),
        A.RandomShadow(),
    ]
)


def corrupt_image(image, filename):
    file_name = os.path.basename(os.path.abspath(filename))
    file_path = os.path.dirname(os.path.abspath(filename))
    for corruption in get_corruption_names():
        corrupted = corrupt(image, severity=5, corruption_name=corruption)
        corrupt_path = file_path.replace(
            "val2017", os.path.join("corruption", corruption)
        )
        if not os.path.exists(corrupt_path):
            os.makedirs(corrupt_path, exist_ok=True)
        cv2.imwrite(os.path.join(corrupt_path, file_name), corrupted)
