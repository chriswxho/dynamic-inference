import torch
import cv2
# import h5py
import numpy as np

from scipy.io import loadmat

import torch.utils.data as data
import torch.nn.functional as F
from torchvision.transforms import Compose

import os

from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet


# class NyuDepthV2(data.Dataset):
#     def __init__(self, datapath, splitpath, split="test", transform=None):

#         self.__image_list = []
#         self.__depth_list = []

#         self.__transform = transform

#         mat = loadmat(splitpath)

#         if split == "train":
#             indices = [ind[0] - 1 for ind in mat["trainNdxs"]]
#         elif split == "test":
#             indices = [ind[0] - 1 for ind in mat["testNdxs"]]
#         else:
#             raise ValueError("Split {} not found.".format(split))

#         with h5py.File(datapath, "r") as f:
#             for ind in indices:
#                 self.__image_list.append(np.swapaxes(f["images"][ind], 0, 2))
#                 self.__depth_list.append(np.swapaxes(f["rawDepths"][ind], 0, 1))

#         self.__length = len(self.__image_list)

#     def __len__(self):
#         return self.__length

#     def __getitem__(self, index):
#         # image
#         image = self.__image_list[index]
#         image = image / 255

#         # depth
#         depth = self.__depth_list[index]

#         # mask; cf. project_depth_map.m in toolbox_nyu_depth_v2 (max depth = 10.0)
#         mask = (depth > 0) & (depth < 10)

#         # sample
#         sample = {}
#         sample["image"] = image
#         sample["depth"] = depth
#         sample["mask"] = mask

#         # transforms
#         if self.__transform is not None:
#             sample = self.__transform(sample)

#         return sample


class BadPixelMetric:
    def __init__(self, threshold=1.25, depth_cap=10):
        self.__threshold = threshold
        self.__depth_cap = depth_cap

    def compute_scale_and_shift(self, prediction, target, mask):
        # system matrix: A = [[a_00, a_01], [a_10, a_11]]
        a_00 = torch.sum(mask * prediction * prediction, (1, 2))
        a_01 = torch.sum(mask * prediction, (1, 2))
        a_11 = torch.sum(mask, (1, 2))

        # right hand side: b = [b_0, b_1]
        b_0 = torch.sum(mask * prediction * target, (1, 2))
        b_1 = torch.sum(mask * target, (1, 2))

        # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
        x_0 = torch.zeros_like(b_0)
        x_1 = torch.zeros_like(b_1)

        det = a_00 * a_11 - a_01 * a_01
        # A needs to be a positive definite matrix.
        valid = det > 0

        x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
        x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

        return x_0, x_1

    def __call__(self, prediction, target, mask):
        # transform predicted disparity to aligned depth
        target_disparity = torch.zeros_like(target)
        target_disparity[mask == 1] = 1.0 / target[mask == 1]

        scale, shift = self.compute_scale_and_shift(prediction, target_disparity, mask)
        prediction_aligned = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        disparity_cap = 1.0 / self.__depth_cap
        prediction_aligned[prediction_aligned < disparity_cap] = disparity_cap

        prediciton_depth = 1.0 / prediction_aligned

        # bad pixel
        err = torch.zeros_like(prediciton_depth, dtype=torch.float)

        err[mask == 1] = torch.max(
            prediciton_depth[mask == 1] / target[mask == 1],
            target[mask == 1] / prediciton_depth[mask == 1],
        )

        err[mask == 1] = (err[mask == 1] > self.__threshold).float()

        p = torch.sum(err, (1, 2)) / torch.sum(mask, (1, 2))

        return 100 * torch.mean(p)


def validate(model, nyu_data, nyu_split):
    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # select device
    device = torch.device("cuda")
    print("device: %s" % device)

    # load network
    model = MidasNet(MODEL_PATH, non_negative=True)
    model.to(device)
    model.eval()

    # get data
    transform = Compose(
        [
            Resize(
                384,
                384,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="upper_bound",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ]
    )

    ds = NyuDepthV2(NYU_DATA_PATH, NYU_SPLIT_PATH, split="test", transform=transform)
    dl = data.DataLoader(
        ds, batch_size=1, num_workers=1, shuffle=False, pin_memory=True
    )

    # validate
    metric = BadPixelMetric()

    loss_sum = 0

    with torch.no_grad():
        for i, batch in enumerate(dl):
            print(f"processing: {i + 1} / {len(ds)}")

            # to device
            for k, v in batch.items():
                batch[k] = v.to(device)

            # run model
            prediction = model.forward(batch["image"])

            # resize prediction to match target
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=batch["mask"].shape[1:],
                mode="bilinear",
                align_corners=False,
            )
            prediction = prediction.squeeze(1)

            loss = metric(prediction, batch["depth"], batch["mask"])
            loss_sum += loss

    print(f"bad pixel: {loss_sum / len(ds):.2f}")


if __name__ == "__main__":
    # download from http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
    NYU_DATA_PATH = "nyu_depth_v2_labeled.mat"

    # download from http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat
    NYU_SPLIT_PATH = "splits.mat"

    # download from https://drive.google.com/file/d/1nqW_Hwj86kslfsXR7EnXpEWdO2csz1cC
    MODEL_PATH = "model.pt"

    # validate
    validate(MODEL_PATH, NYU_DATA_PATH, NYU_SPLIT_PATH)
