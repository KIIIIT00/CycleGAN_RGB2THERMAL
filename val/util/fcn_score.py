"""
FCN-Scoreを計算
"""

import torch, torchvision
from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet50
from PIL import Image
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import albumentations as A
import cv2
import os

# 結果のフォルダ
RESULT_FOLDER = "./results/rgbtothermal300_cyclegan/rgbtothermal300_cyclegan/test_latest/images/"
# ファイル名の一覧を取得
files = os.listdir(RESULT_FOLDER)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 学習済みのFCNモデルを取得
model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
model.to(device)

real = 'real_B'
fake = 'fake_B'

real_name_list = [file for file in files if real in file]
fake_name_list = [file for file in files if fake in file]
real_name_list.sort()
fake_name_list.sort()

num = 0
score_laberl0 = 0
score_laberl1 = 0
for real, fake in zip(real_name_list, fake_name_list):
    real_img = cv2.imread(os.path.join(RESULT_FOLDER, real))
    fake_img = cv2.imread(os.path.join(RESULT_FOLDER, fake))

    transform = A.Compose([
    A.Resize(256,256),
    A.Normalize()
    ])
    
    real_t =  torch.tensor(
    transform(image = real_img)["image"]
    ).transpose(0,2).transpose(1,2).unsqueeze(0)

    fake_t =  torch.tensor(
    transform(image=fake_img)["image"]
    ).transpose(0,2).transpose(1,2).unsqueeze(0)

    output_real = model(real_t.to(device))
    output_fake = model(fake_t.to(device))

    seg_real = output_real["out"].squeeze().detach().argmax(0).cpu().numpy()
    seg_fake = output_fake["out"].squeeze().detach().argmax(0).cpu().numpy()

    score_laberl0 += accuracy_score(seg_real.flatten(), seg_fake.flatten())
    score_laberl1 += accuracy_score(seg_real[seg_real > 0].flatten(), seg_fake[seg_real > 0].flatten())
 

    num += 1

    

plt.subplot(2,2,1)
plt.title("real")
plt.imshow(real_img)
plt.axis("off")
plt.subplot(2,2,2)
plt.title("fake")
plt.imshow(fake_img)
plt.axis("off")

plt.subplot(2,2,3)
plt.title("segreal")
plt.imshow(seg_real, vmin=0, vmax=21)
plt.colorbar()
plt.axis("off")
plt.subplot(2,2,4)
plt.title("segfake")
plt.imshow(seg_fake, vmin=0, vmax=21)
plt.colorbar()
plt.axis("off")
print("FCN Score(0):{}'".format(score_laberl0 / num))
print("FCN Score(1):{}'".format(score_laberl1 / num))

plt.tight_layout()
plt.show()

