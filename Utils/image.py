# _*_ coding: utf-8 _*_
# @author   : 王福森
# @time     : 2021/12/9 19:59
# @File     : image.py.py
# @Software : PyCharm
import torch
import torch.nn.functional as F

def tensor_spilt(img_tensor):
    _, _, h, w = img_tensor.size()
    m, n = int(w/384), int(h/384)
    for i in range(0, m):
        for j in range(0, n):
            if i == 0 and j == 0:
                img_return = img_tensor[:, :, j*384 : (j+1)*384, i*384 : (i+1)*384]
            else:
                crop_img = img_tensor[:, :, j*384 : (j+1)*384, i*384 : (i+1)*384]
                img_return = torch.cat([img_return, crop_img], 0)
    return img_return

def tensor_divideByfactor(img_tensor, factor=32):
    _, _, h, w = img_tensor.size()
    h, w = int(h//factor*factor), int(w//factor*factor)
    img_tensor = F.interpolate(img_tensor, (h, w), mode='bilinear', align_corners=True)

    return img_tensor

def cal_new_tensor(img_tensor, min_size=256):
    _, _, h, w = img_tensor.size()
    if min(h, w) < min_size:
        ratio_h, ratio_w = min_size / h, min_size / w
        if ratio_h >= ratio_w:
            img_tensor = F.interpolate(img_tensor, (min_size, int(min_size / h * w)), mode='bilinear', align_corners=True)
        else:
            img_tensor = F.interpolate(img_tensor, (int(min_size / w * h), min_size), mode='bilinear', align_corners=True)
    return img_tensor