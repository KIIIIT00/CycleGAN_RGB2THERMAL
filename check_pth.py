import torch
from models import networks
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

def load_network(MODEL_PATH, device):
    state_dict = torch.load(MODEL_PATH, map_location = str(device))
    for key in list(state_dict.keys()):
        print("Key:",key)

def adjust_netG(netG, state_dict):
    netG.load_state_dict(state_dict, strict=False)
    netG.eval()
    return netG


MODEL_NAME = 'Scene2ver2_500_lambda10.5'
MODEL_PATH = f'./checkpoints/{MODEL_NAME}/latest_net_G.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_params = torch.load(MODEL_PATH, map_location=str(device))
netG = networks.define_G(input_nc=3,
                         output_nc=3,
                         ngf=64,
                         netG='resnet_9blocks')
print(netG)
print("Model:", dict(netG.named_modules()).keys())
# for param in netG.parameters():
#     print(param.shape)
# load_network(MODEL_PATH, device)
# netG_eval = adjust_netG(netG, model_params)
# print(type(netG_eval))