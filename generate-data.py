import os
import sys
import tempfile
import shutil
from glob import glob
import logging
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.transforms import (
    Compose,
    LoadNiftid,
    AsChannelFirstd,
    ScaleIntensityd,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ToTensord,
)
from monai.data import create_test_image_3d, list_data_collate
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.visualize import plot_2d_or_3d_image

from catalyst import dl

monai.config.print_config()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# create a temporary directory and 40 random image, mask paris
tempdir = '/data/mialab/users/washbee/tempdata'
print("generating synthetic data to {tempdir} (this may take a while)")
for i in range(40):
    im, seg = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)
    n = nib.Nifti1Image(im, np.eye(4))
    #print (os.path.join(tempdir, f"img{i:d}.nii.gz"))
    #break
    nib.save(n, os.path.join(tempdir, f"img{i:d}.nii.gz"))
    n = nib.Nifti1Image(seg, np.eye(4))
    nib.save(n, os.path.join(tempdir, f"seg{i:d}.nii.gz"))
images = sorted(glob(os.path.join(tempdir, "img*.nii.gz")))
segs = sorted(glob(os.path.join(tempdir, "seg*.nii.gz")))