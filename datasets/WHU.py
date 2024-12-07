import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from skimage.draw import polygon2mask
from datasets.tools import ResizeAndPad, soft_transform, collate_fn, collate_fn_soft, collate_fn_


class WHUDataset(Dataset):
    def __init__(self, cfg, root_dir, annotation_file, rate=(5, 1), transform=None, training=False, if_self_training=False, gen_pt = False):
        self.cfg = cfg
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(annotation_file)
        self.image_ids = sorted(list(self.coco.imgs.keys()))
        self.training = training
        self.gen_pt = gen_pt

        # # Filter out image_ids without any annotations
        self.image_ids = [
            image_id
            for image_id in self.image_ids
            if len(self.coco.getAnnIds(imgIds=image_id)) > 0
        ]

        self.if_self_training = if_self_training

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        if self.training or self.gen_pt:
            image_path = os.path.join(self.root_dir,'train/image', image_info["file_name"])
        else:
            image_path = os.path.join(self.root_dir,'val/image', image_info["file_name"])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        bboxes = []
        masks = []
        categories = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            bboxes.append([x, y, x + w, y + h])
            mask = self.coco.annToMask(ann)
            masks.append(mask)
            categories.append(ann["category_id"])

        if self.if_self_training:
            image_weak, bboxes_weak, masks_weak, image_strong = soft_transform(image, bboxes, masks, categories)

            if self.transform:
                image_weak, masks_weak, bboxes_weak = self.transform(image_weak, masks_weak, np.array(bboxes_weak))
                image_strong = self.transform.transform_image(image_strong)

            bboxes_weak = np.stack(bboxes_weak, axis=0)
            masks_weak = np.stack(masks_weak, axis=0)
            return image_weak, image_strong, torch.tensor(bboxes_weak), torch.tensor(masks_weak).float(), image_path

        elif self.cfg.visual:
            origin_image = image
            origin_bboxes = bboxes
            origin_masks = masks
            if self.transform:
                padding, image, masks, bboxes = self.transform(image, masks, np.array(bboxes), True)

            bboxes = np.stack(bboxes, axis=0)
            masks = np.stack(masks, axis=0)
            origin_bboxes = np.stack(origin_bboxes, axis=0)
            origin_masks = np.stack(origin_masks, axis=0)
            return image_id, padding, origin_image, origin_bboxes, origin_masks, image, torch.tensor(bboxes), torch.tensor(masks).float()

        else:
            if self.transform:
                image, masks, bboxes = self.transform(image, masks, np.array(bboxes))
            bboxes = np.stack(bboxes, axis=0)
            masks = np.stack(masks, axis=0)
            return image, torch.tensor(bboxes), torch.tensor(masks).float(),image_path

def load_datasets(cfg, img_size):
    transform = ResizeAndPad(img_size)
    train = WHUDataset(
        cfg,
        root_dir=cfg.datasets.WHU.root_dir,
        annotation_file=cfg.datasets.WHU.annotation_file_train,
        transform=transform,
        training=True,
    )
    train_dataloader = DataLoader(
        train,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )

    val = WHUDataset(
        cfg,
        root_dir=cfg.datasets.WHU.root_dir,
        annotation_file=cfg.datasets.WHU.annotation_file_val,
        transform=transform,
    )
    val_dataloader = DataLoader(
        val,
        batch_size=cfg.val_batchsize,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    return train_dataloader, val_dataloader


def load_datasets_soft(cfg, img_size, return_pt = False):
    transform = ResizeAndPad(img_size)

    soft_train = WHUDataset(
        cfg,
        root_dir=cfg.datasets.WHU.root_dir,
        annotation_file=cfg.datasets.WHU.annotation_file_train,
        transform=transform,
        training=True,
        if_self_training=True,
    )
    soft_train_dataloader = DataLoader(
        soft_train,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_soft,
    )

    val = WHUDataset(
        cfg,
        root_dir=cfg.datasets.WHU.root_dir,
        annotation_file=cfg.datasets.WHU.annotation_file_val,
        transform=transform,
    )
    val_dataloader = DataLoader(
        val,
        batch_size=cfg.val_batchsize,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )
    
    if return_pt:
        pt = WHUDataset(
            cfg,
            root_dir=cfg.datasets.WHU.root_dir,
            annotation_file=cfg.datasets.WHU.annotation_file_train,
            transform=transform,
            gen_pt = return_pt,
        )
        pt_dataloader = DataLoader(
            pt,
            batch_size=cfg.val_batchsize,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
        )
        return soft_train_dataloader, val_dataloader, pt_dataloader
    else:
        return soft_train_dataloader, val_dataloader

def load_datasets_visual(cfg, img_size):
    transform = ResizeAndPad(img_size)
    val = WHUDataset(
        cfg,
        root_dir=cfg.datasets.WHU.root_dir,
        annotation_file=cfg.datasets.WHU.annotation_file_val,
        transform=transform,
    )
    val_dataloader = DataLoader(
        val,
        batch_size=cfg.val_batchsize,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn_,
    )
    return val_dataloader
