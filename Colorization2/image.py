import torch
import cv2
import os
from utils import utils_image as util
from models.network_rrdbnet import RRDBNet as net

img_path = 'images/1.jpg' #이미지 경로

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4)
model.load_state_dict(torch.load(os.path.join('model', 'BSRGAN.pth')), strict=True)
model = model.to(device)
model.eval()

with torch.no_grad():
    img = cv2.imread(img_path)

    img_L = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    img_L = util.uint2tensor(img_L)
    img_L = img_L.to(device)

    img_E = model(img_L)

    img_E = util.tensor2uint(img_E)
    util.imsave(img_E, os.path.splitest(img_path)[0] + '_result.png')