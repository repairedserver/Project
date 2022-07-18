import os
import torch
import cv2

import utils.utils_image
from utils import utils_image as util
from models.network_rrdbnet import RRDBNet as net

video_path = 'videos/LotteBigbar.mp4'
model_path = os.path.join('model', 'BSRGAN.pth')

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4)

model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
for a, b in model.named_parameters():
    b.requires_grad = False

model = model.to(device)

cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('%s_output.mp4' %(video_path.split('.')[0]), fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 4), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 4)))

n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
i = 0

print(n_frames)
cap.set(cv2.CAP_PROP_POS_FRAMES, i)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img_L = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    img_L = utils.uint2tensor4(img_L)
    img_L = utils.to(device)