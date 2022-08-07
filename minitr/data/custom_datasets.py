from detectron2.data.datasets.coco import register_coco_instances
import os
from alfred.modules.data.view_coco import get_coco_info


def register_ap10k_datasets():
    # AP-10k dataset
    DATASET_ROOT = "./datasets/ap10k"
    if not os.path.exists(DATASET_ROOT):
        return
    ANN_ROOT = os.path.join(DATASET_ROOT, "annotations")
    TRAIN_PATH = os.path.join(DATASET_ROOT, "data")
    VAL_PATH = os.path.join(DATASET_ROOT, "data")
    TRAIN_JSON1 = os.path.join(ANN_ROOT, "ap10k-train-split1.json")
    TRAIN_JSON2 = os.path.join(ANN_ROOT, "ap10k-train-split2.json")
    TRAIN_JSON3 = os.path.join(ANN_ROOT, "ap10k-train-split3.json")
    VAL_JSON1 = os.path.join(ANN_ROOT, "ap10k-val-split1.json")
    VAL_JSON2 = os.path.join(ANN_ROOT, "ap10k-val-split2.json")
    VAL_JSON3 = os.path.join(ANN_ROOT, "ap10k-val-split3.json")
    keypoint_names = [
        "left_eye",
        "right_eye",
        "nose",
        "neck",
        "root_of_tail",
        "left_shoulder",
        "left_elbow",
        "left_front_paw",
        "right_shoulder",
        "right_elbow",
        "right_front_paw",
        "left_hip",
        "left_knee",
        "left_back_paw",
        "right_hip",
        "right_knee",
        "right_back_paw",
    ]
    keypoint_flip_map = {
        "left_eye": "right_eye",
        "left_shoulder": "right_shoulder",
        "left_elbow": "right_elbow",
        "left_front_paw": "right_front_paw",
        "left_hip": "right_hip",
        "left_knee": "right_knee",
        "left_back_paw": "right_back_paw",
    }
    info = get_coco_info(VAL_JSON1)
    class_names = info['class_names']
    skeleton = info['skeleton_dict'][class_names[0]]
    meta_data = {
        "keypoint_names": keypoint_names,
        "keypoint_flip_map": keypoint_flip_map,
        "thing_classes": class_names,
        "skeleton": skeleton
    }
    register_coco_instances(
        "keypoints_ap10k_train1", meta_data, TRAIN_JSON1, TRAIN_PATH
    )
    register_coco_instances(
        "keypoints_ap10k_train2", meta_data, TRAIN_JSON2, TRAIN_PATH
    )
    register_coco_instances(
        "keypoints_ap10k_train3", meta_data, TRAIN_JSON3, TRAIN_PATH
    )
    register_coco_instances("keypoints_ap10k_val1", meta_data, VAL_JSON1, TRAIN_PATH)
    register_coco_instances("keypoints_ap10k_val2", meta_data, VAL_JSON2, TRAIN_PATH)
    register_coco_instances("keypoints_ap10k_val3", meta_data, VAL_JSON3, TRAIN_PATH)
