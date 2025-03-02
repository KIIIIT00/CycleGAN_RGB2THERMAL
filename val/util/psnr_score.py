"""
PSNRを計算する
参考：https://qiita.com/Daiki_P/items/94662fd340aa0381b323

評価指標：https://jp.mathworks.com/help/vision/ref/psnr.html
"""

import cv2
import numpy as np
import math
import os
import lpips
import torch
from skimage.metrics import structural_similarity
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance

# 結果のフォルダ
RESULT_FOLDER = "./results/rgbtothermal300_cyclegan/rgbtothermal300_cyclegan/test_latest/images/"
# ファイル名の一覧を取得
files = os.listdir(RESULT_FOLDER)
real = 'real_B'
fake = 'fake_B'

real_name_list = [file for file in files if real in file]
fake_name_list = [file for file in files if fake in file]
real_name_list.sort()
fake_name_list.sort()

def calculate_mse(real, fake):
    """
    MSEを計算をする

    Parameters
    ------------
    real : ndarray
        正解データ
    fake : ndarray
        出力データ

    Returns
    ------------
    MSE : ndarray
        MSE
    """
    return np.mean((real - fake) ** 2)

def calculate_rmse(real, fake):
    """
    RMSEを計算をする

    Parameters
    ------------
    real : ndarray
        正解データ
    fake : ndarray
        出力データ

    Returns
    ------------
    RMSE : ndarray
        RMSE
    """
    return math.sqrt(calculate_mse(real, fake))

def calcualte_psnr(real, fake):
    """
    PSNRを計算をする

    Parameters
    ------------
    real : ndarray
        正解データ
    fake : ndarray
        出力データ

    Returns
    ------------
    PSNR : ndarray
        PSNR
    """
    mse = calculate_mse(real, fake)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def calculate_lpips(real, fake, loss_fn):
    """
    LPIPSを計算をする
    
    Parameters
    ------------
    real : ndarray
    fake : ndarray
    
    Returns
    lpips : ndarray
    """
    real_tensor = torch.from_numpy(real).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    fake_tensor = torch.from_numpy(fake).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    
    lpips_value = loss_fn(real_tensor, fake_tensor).item()
    return lpips_value


# LPIPS
lpips_net_alex = 'alex' # slect 'alex' or 'vgg' or 'squeeze'
loss_fn_alex = lpips.LPIPS(net=lpips_net_alex)
loss_fn_vgg = lpips.LPIPS(net='vgg')
loss_fn_squeeze = lpips.LPIPS(net='squeeze')

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# kid_metric = KernelInceptionDistance(subset_size=50).to(device)
# fid_metric = FrechetInceptionDistance().to(device)

num = len(real_name_list)
mse = 0
rmse = 0
psnr = 0
ssim = 0
lpips_score_alex = 0
lpips_score_vgg = 0
lpips_score_squeeze = 0

# 画像のリスト
# real_images = []
# fake_images = []

for real, fake in zip(real_name_list, fake_name_list):
    real_img = cv2.imread(os.path.join(RESULT_FOLDER, real))
    fake_img = cv2.imread(os.path.join(RESULT_FOLDER, fake))
    
    mse += calculate_mse(real_img, fake_img)
    rmse += calculate_rmse(real_img, fake_img)
    psnr += calcualte_psnr(real_img, fake_img)
    ssim_index, ss_image= structural_similarity(real_img, fake_img,full=True, win_size = 7, channel_axis=-1)
    ssim += ssim_index
    lpips_score_alex += calculate_lpips(real_img, fake_img, loss_fn_alex)
    lpips_score_vgg += calculate_lpips(real_img, fake_img, loss_fn_vgg)
    lpips_score_squeeze += calculate_lpips(real_img, fake_img, loss_fn_squeeze)
    
    
    # Tensor Translation
    # real_tensor = torch.from_numpy(real_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    # fake_tensor = torch.from_numpy(fake_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    # real_images.append(real_tensor)
    # fake_images.append(fake_tensor)

# real_images = torch.cat(real_images).to(torch.uint8).to(device)
# fake_images = torch.cat(fake_images).to(torch.uint8).to(device)
# kid_metric.update(real_images, real=True)
# kid_metric.update(fake_images, real=False)
# fid_metric.update(real_images, real=True)
# fid_metric.update(fake_images, real=False)

# kid_mean, kid_std = kid_metric.compute()
# fid_score = fid_metric.compute()

# 表示の桁数を指定
precision = 6

print(f"MSE: {mse / len(real_name_list):.{precision}f}")
print(f"RMSE: {rmse / len(real_name_list):.{precision}f}")
print(f"PSNR: {psnr / len(real_name_list):.{precision}f}")
print(f"SSIM: {ssim / len(real_name_list):.{precision}f}")
print(f"LPIPS(Alex): {lpips_score_alex / len(real_name_list):.{precision}f}")
print(f"LPIPS(VGG): {lpips_score_vgg / len(real_name_list):.{precision}f}")
print(f"LPIPS(Squeeze): {lpips_score_squeeze / len(real_name_list):.{precision}f}")
# print(f"KID (Mean): {kid_mean.item():.{precision}f}")
# print(f"KID (Std): {kid_std.item():.{precision}f}")
# print(f"FID: {fid_score.item():.{precision}f}")