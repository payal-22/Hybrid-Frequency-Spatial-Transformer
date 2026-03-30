import os
import random
import numpy as np
import cv2

from torch.utils.data import Dataset
from utils import hwc_to_chw, read_img


class VideoFrameLoader(Dataset):
    def __init__(self, data_dir, sub_dir, mode, size=256, edge_decay=0, only_h_flip=False):
        assert mode in ['train', 'valid', 'test']

        self.mode = mode
        self.size = size
        self.edge_decay = edge_decay
        self.only_h_flip = only_h_flip

        self.root_dir = os.path.join(data_dir, sub_dir)
        self.video_names = sorted(os.listdir(os.path.join(self.root_dir, 'GT')))
        self.video_num = len(self.video_names)

        # 提取所有视频的帧列表进行训练
        self.frames = []
        for video_name in self.video_names:
            cap = cv2.VideoCapture(os.path.join(self.root_dir, 'INPUT', video_name))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frames.extend([(video_name, frame) for frame in range(frame_count)])
            cap.release()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        video_name, frame_idx = self.frames[idx]

        # 使用cv2读取特定帧
        cap = cv2.VideoCapture(os.path.join(self.root_dir, 'INPUT', video_name))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, source_img = cap.read()
        cap.release()

        if not ret:
            raise IOError("Couldn't read frame number {} for video {}".format(frame_idx, video_name))

        # 读取相应的“ground truth”帧和先验帧，假设它们以相同的帧速率和数量存在。
        cap_gt = cv2.VideoCapture(os.path.join(self.root_dir, 'GT', video_name))
        cap_gt.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret_gt, target_img = cap_gt.read()
        cap_gt.release()

        if not ret_gt:
            raise IOError("Couldn't read GT frame number {} for video {}".format(frame_idx, video_name))

        cap_dc = cv2.VideoCapture(os.path.join(self.root_dir, 'DC', video_name))
        cap_dc.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret_dc, prior_img = cap_dc.read()
        cap_dc.release()

        if not ret_dc:
            raise IOError("Couldn't read DC frame number {} for video {}".format(frame_idx, video_name))

        # 对帧进行缩放
        source_img = source_img.astype('float32') / 255.0 * 2 - 1
        target_img = target_img.astype('float32') / 255.0 * 2 - 1
        prior_img = prior_img.astype('float32') / 255.0 * 2 - 1

        # 帧的数据增强和配准...

        # 返回字典包含处理后的帧和其他信息
        return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img), 'prior': hwc_to_chw(prior_img),
                'filename': video_name + "_frame_{}".format(frame_idx)}

# 辅助函数 `hwc_to_chw` 需要针对处理的帧数进行相应的转换。