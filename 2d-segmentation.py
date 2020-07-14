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
from monai.data import create_test_image_2d, list_data_collate
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
    im, seg = create_test_image_2d(128, 128, num_seg_classes=1, channel_dim=-1)
    print(type(im))
    print(im.shape)
    
    n = nib.Nifti1Image(im, np.eye(4))
    nib.save(n, os.path.join(tempdir, f"img{i:d}.nii.gz"))
    print (os.path.join(tempdir, f"img{i:d}.nii.gz"))
    n = nib.Nifti1Image(seg, np.eye(4))
    nib.save(n, os.path.join(tempdir, f"seg{i:d}.nii.gz"))
images = sorted(glob(os.path.join(tempdir, "img*.nii.gz")))
segs = sorted(glob(os.path.join(tempdir, "seg*.nii.gz")))

train_files = [{"img": img, "seg": seg} for img, seg in zip(images[:20], segs[:20])]
val_files = [{"img": img, "seg": seg} for img, seg in zip(images[-20:], segs[-20:])]

# define transforms for image and segmentation
train_transforms = Compose(
    [
        LoadNiftid(keys=["img", "seg"]),
        AsChannelFirstd(keys=["img", "seg"], channel_dim=-1),
        ScaleIntensityd(keys=["img", "seg"]),
        RandCropByPosNegLabeld(
            keys=["img", "seg"], label_key="seg", spatial_size=[96, 96, 96], pos=1, neg=1, num_samples=4
        ),
        RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 2]),
        ToTensord(keys=["img", "seg"]),
    ]
)
val_transforms = Compose(
    [
        LoadNiftid(keys=["img", "seg"]),
        AsChannelFirstd(keys=["img", "seg"], channel_dim=-1),
        ScaleIntensityd(keys=["img", "seg"]),
        ToTensord(keys=["img", "seg"]),
    ]
)

# define dataset, data loader
check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
# use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
check_loader = DataLoader(check_ds, batch_size=2, num_workers=4, collate_fn=list_data_collate)
check_data = monai.utils.misc.first(check_loader)
print(check_data["img"].shape, check_data["seg"].shape)

# create a training data loader
train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
# use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
train_loader = DataLoader(
    train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    collate_fn=list_data_collate,
    pin_memory=torch.cuda.is_available(),
)
# create a validation data loader
val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate)

# create UNet, DiceLoss and Adam optimizer
# device = torch.device("cuda:0")  # you don't need device, because Catalyst uses autoscaling
model = monai.networks.nets.UNet(
    dimensions=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)
loss_function = monai.losses.DiceLoss(sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-3)

dice_metric = DiceMetric(include_background=True, to_onehot_y=False, sigmoid=True, reduction="mean")

class MonaiSupervisedRunner(dl.SupervisedRunner):

  def forward(self, batch):
    if self.is_train_loader:
      output = {self.output_key: self.model(batch[self.input_key])}
    elif self.is_valid_loader:
      roi_size = (96, 96, 96)
      sw_batch_size = 4
      output = {self.output_key: sliding_window_inference(batch[self.input_key], roi_size, sw_batch_size, self.model)}
    elif self.is_infer_loader:
      roi_size = (96, 96, 96)
      sw_batch_size = 4
      batch = self._batch2device(batch, self.device)
      output = {self.output_key: sliding_window_inference(batch[self.input_key], roi_size, sw_batch_size, self.model)}
      output = {**output, **batch}
    return output

runner = MonaiSupervisedRunner(input_key="img", input_target_key="seg", output_key="logits")  # you can also specify `device` here
runner.train(
    loaders={"train": train_loader, "valid": val_loader},
    model=model,
    criterion=loss_function,
    optimizer=optimizer,
    num_epochs=6, logdir="./logs", 
    main_metric="dice_metric", minimize_metric=False,
    verbose=False, timeit=True,  # let's use minimal logs, but with time checkers
    callbacks={
        "loss": dl.CriterionCallback(input_key="seg", output_key="logits"),
        "periodic_valid": dl.PeriodicLoaderCallback(valid=2),
        "dice_metric": dl.MetricCallback(prefix="dice_metric", metric_fn=dice_metric, input_key="seg", output_key="logits")
    },
    load_best_on_end=True,  # user-friendly API :)
)