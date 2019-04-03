#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import time
import argparse
import glob
from PIL import Image
import numpy as np
import cv2

import torch
import torch.nn.functional as F
from torchvision import transforms

from models.resnet import ResNet18


parser = argparse.ArgumentParser()
parser.add_argument("--wts", dest="weights", default=None, type=str, help="tht model weights.")
parser.add_argument("--output-dir", dest="output_dir", help="directory for visualization images.",
                    default="/fastai/tmp/infer",
                    type=str)
parser.add_argument("img_or_folder", type=str, help="an image or a folder to test.")


def returnCAM(feature_conv, weight_softmax, class_idx):
    upsample_size = (256, 256)
    bz, nc, h, w = feature_conv.shape
    cam = weight_softmax[class_idx].dot(feature_conv.reshape(nc, h * w))
    cam = cam.reshape(h, w)
    cam -= np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    return cv2.resize(cam_img, upsample_size)


if __name__ == "__main__":
    args = parser.parse_args()
    model = ResNet18(label_number=37)
    model.load_state_dict(torch.load(args.weights))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    if os.path.isdir(args.img_or_folder):
        im_list = glob.iglob(os.path.join(args.img_or_folder, '*.jpg'))
    else:
        im_list = [args.img_or_folder]

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    feature_blobs = []

    def hook_feature(module, input, output):
        feature_blobs.append(output.data.cpu().numpy())

    model._modules.get("stage3").register_forward_hook(hook_feature)
    params = list(model.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
    for i, im_name in enumerate(im_list):
        out_name = os.path.join(args.output_dir, os.path.basename(im_name))
        print("Processing {} -> {}".format(im_name, out_name))
        since = time.time()
        im = Image.open(im_name)
        im_tensor = transform(im).to(device)
        logit = model(im_tensor.unsqueeze(0)).cpu()
        h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs = probs.numpy()
        idx = idx.numpy()
        cam = returnCAM(feature_blobs[0], weight_softmax, idx[0])
        img = cv2.imread(im_name)
        height, width, _ = img.shape
        heatmap = cv2.applyColorMap(cv2.resize(cam, (width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.3 + img * 0.5
        cv2.imwrite(out_name, result)
