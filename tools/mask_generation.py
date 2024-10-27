import cv2
import numpy as np
import supervision as sv
import os
import argparse
import warnings
from collections import OrderedDict
from tqdm import tqdm
import sys

import torch
import torchvision

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor


GDINO_CONFIG_RELATIVE_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_CKPT_RELATIVE_PATH = "groundingdino_swint_ogc.pth"
SAM_ENCODER_VERSION = "vit_h"
SAM_CKPT_RELATIVE_PATH = "sam_vit_h_4b8939.pth"
SPEC_CLASS_BOX_MAX_NUMS = {
        "brow": 2,
        "eyebrow": 2,
        "eye": 2,
        "ear": 2,
        "wheel": 2,
        "window": 3,
    }
    

def check_mask_existence(image_dir):
    assert os.path.isdir(image_dir)
    mask_dir = os.path.join(image_dir, 'masks')
    if not os.path.isdir(mask_dir):
        return False
    image_basenames = set(
        os.path.splitext(f)[0] 
        for f in os.listdir(image_dir)
        if os.path.isfile(os.path.join(image_dir, f)) and f.endswith(('.png', '.jpg', '.jpeg'))
    )
    mask_basenames = set(
        os.path.splitext(f)[0] 
        for f in os.listdir(mask_dir)
        if os.path.isfile(os.path.join(mask_dir, f)) and f.endswith(('.png', '.jpg', '.jpeg'))
    )
    if image_basenames == mask_basenames:
        return True
    else:
        return False


# Prompting SAM with detected boxes
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


def get_class_names(image_dir, no_redundant=False, class_nums=-1):
    filenames = [
        f for f in sorted(os.listdir(image_dir))
        if os.path.isfile(os.path.join(image_dir, f)) and f.endswith(('.png', '.jpg', '.jpeg'))
    ]
    class_names = [
        filename.split('_')[1] for filename in filenames
    ]
    if no_redundant:
        class_names = list(OrderedDict.fromkeys(class_names))
    if class_nums != -1:
        class_names = class_names[:class_nums]

    return class_names, filenames


def save_mask_image(mask_image, image_name, mask_dir):
    mask_image = np.where(mask_image, 255, 0).astype(np.uint8)
    mask_image = np.transpose(mask_image, (1, 2, 0))
    cv2.imwrite(os.path.join(mask_dir, image_name+".png"), mask_image)


def generate_masks(args, grounding_dino_model, sam_predictor, image_dir, save_logs=True, check_existence=False):

    tqdm.write("-" * 50)
    tqdm.write(f"Processing: {image_dir}")

    mask_dir = os.path.join(image_dir, 'masks')
    mask_others_dir = os.path.join(mask_dir, "others")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(mask_others_dir, exist_ok=True)

    if check_existence and check_mask_existence(image_dir):
        tqdm.write("Masks alreadly exist.")
        return

    # warnings.filterwarnings("ignore")

    # get class names for segmentation
    class_names, filenames = get_class_names(image_dir)
    assert len(class_names) > 1
    seg_class_names = []
    concept_name = class_names[0]
    for name_i in class_names:
        seg_class_names.append([name_i])
        if name_i == concept_name:
            for name_j in class_names:
                if name_j != concept_name and name_j not in seg_class_names[-1]:
                    seg_class_names[-1].append(name_j)
    tqdm.write(f"seg_class_names: {seg_class_names}")

    for i, filename in enumerate(filenames):

        file_path = os.path.join(image_dir, filename)
        image_name = os.path.splitext(filename)[0]
        image = cv2.imread(file_path)

        classes = seg_class_names[i]

        # detect objects
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=classes,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold
        )

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()

        # NMS post process
        # print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            args.nms_threshold
        ).numpy().tolist()
        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        # print(f"After NMS: {len(detections.xyxy)} boxes")
        
        # get top-K boxes
        topk_nums = [
            SPEC_CLASS_BOX_MAX_NUMS[c] if c in SPEC_CLASS_BOX_MAX_NUMS else 1
            for c in classes
        ]
        topk_idx = []
        for j, id in enumerate(set(detections.class_id)):
            k = topk_nums[j]
            id_idx = np.where(detections.class_id == id)[0]
            id_confidence = detections.confidence[id_idx]
            topk_idx.append(id_idx[np.argsort(id_confidence)[-k:]])
        topk_idx = np.hstack(topk_idx)
        detections.xyxy = detections.xyxy[topk_idx]
        detections.confidence = detections.confidence[topk_idx]
        detections.class_id = detections.class_id[topk_idx]

        # convert detections to masks
        detections.mask = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )
        
        # save the mask image
        masks = []
        for j, id in enumerate(set(detections.class_id)):
            id_idx = np.where(detections.class_id == id)[0]
            masks.append(np.any(detections.mask[id_idx], axis=0, keepdims=True))
        masks = np.concatenate(masks)
        if masks.shape[0] > 1:
            masks[1:, :, :] = np.logical_not(masks[1:, :, :])
            mask_image_w_comp = np.expand_dims(masks[0], axis=0)  # mask w/ component
            mask_image = np.all(masks, axis=0, keepdims=True)  # mask wo/ component
            save_mask_image(mask_image, image_name, mask_dir)
            save_mask_image(mask_image_w_comp, image_name, mask_others_dir)
        else:
            mask_image = masks
            save_mask_image(mask_image, image_name, mask_dir)

        # save logs
        if save_logs:
            # init log dir
            log_dir = os.path.join(image_dir, 'logs')
            os.makedirs(log_dir, exist_ok=True)

            # get the annotated image of grounding dino 
            labels = [
                f"{classes[class_id]} {confidence:0.2f}" 
                for _, _, confidence, class_id, _, _ 
                in detections]
            annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)
            # cv2.imwrite(os.path.join(log_dir, image_name+"_gdino.jpg"), annotated_frame)

            # get the annotated image of grounding-SAM
            box_annotator = sv.BoxAnnotator()
            mask_annotator = sv.MaskAnnotator()
            annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
            annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
            cv2.imwrite(os.path.join(log_dir, image_name+"_gsam.jpg"), annotated_image)

        # warnings.resetwarnings()

def get_gdino_and_sam_model(args, device):
    # GroundingDINO config and checkpoint
    gdino_config_path = os.path.join(args.gsam_repo_dir, GDINO_CONFIG_RELATIVE_PATH)
    gdino_ckpt_path = os.path.join(args.gsam_repo_dir, GDINO_CKPT_RELATIVE_PATH)

    # Segment-Anything checkpoint
    sam_ckpt_path = os.path.join(args.gsam_repo_dir, SAM_CKPT_RELATIVE_PATH)

    # Building GroundingDINO inference model
    grounding_dino_model = Model(model_config_path=gdino_config_path, model_checkpoint_path=gdino_ckpt_path)

    # Building SAM Model and SAM Predictor
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=sam_ckpt_path)
    sam.to(device=device)
    sam.eval()
    sam_predictor = SamPredictor(sam)

    return grounding_dino_model, sam_predictor

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Mask Generation with G-SAM", add_help=True)

    parser.add_argument(
        "--gsam_repo_dir", default="Grounded-Segment-Anything",
        type=str, help="dir to gsam repo",
        # required=True, 
    )
    parser.add_argument(
        "--dataset_dir", default="dataset_test", # ./tailorbench
        type=str, help="dir to the dataset",
        # required=True, 
    )

    parser.add_argument("--box_threshold", type=float, default=0.25, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--nms_threshold", type=float, default=0.8, help="nms threshold")

    args = parser.parse_args()

    print("Start geneartive masks for dataset ...")
    print(f"Dataset directory: {args.dataset_dir}")

    warnings.filterwarnings("ignore")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grounding_dino_model, sam_predictor = get_gdino_and_sam_model(args, device)

    subfolders = [
            f for f in sorted(os.listdir(args.dataset_dir))
            if os.path.isdir(os.path.join(args.dataset_dir, f))
        ]

    for subfolder in tqdm(subfolders, file=sys.stdout, desc="Pair Progress"):
        subfolder_dir = os.path.join(args.dataset_dir, subfolder)
        generate_masks(args, grounding_dino_model, sam_predictor, subfolder_dir, check_existence=True)
