import os #운영체제 제공 기능 모듈
import torch #pytorch
import cv2 #opencv-python

import utils.utils_image
from utils import utils_image as util #유틸 모듈
from models.network_rrdbnet import RRDBNet as net #models 폴더에서 RRDBNet 임포트

#데이터셋
#https://github.com/cszn/KAIR/releases/download/v1.0/BSRGAN

video_path = 'videos/LotteBigbar.mp4' #화질 향상시킬 동영상 경로지정
model_path = os.path.join('model', 'BSRGAN.pth') #model 폴더에서 데이터셋 넣기

#학습시킬 프로세서 cuda:GPU, cpu:CPU
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4)

model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
for a, b in model.named_parameters():
    b.requires_grad = False

model = model.to(device) #GPU가 있다면 GPU로, 없다면 CPU로 러닝

cap = cv2.VideoCapture(video_path)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

#이미지를 {name}_output.mp4 형태로 저장
out = cv2.VideoWriter('%s_output.mp4' %(video_path.split('.')[0]), fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WDIDTH) * 4), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 4)))

n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
i = 0

print(n_frames)
cap.set(cv2.CAP_PROP_POS_FRAMES, i)

#영상 프레임당 화질 향상
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img_L = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB) #4채널을 3채널로 변경
    img_L = util.uint2tensor4(img_L) #텐서 형태로 변환
    img_L = img_L.to(device) #GPU가 있다면 GPU로, 없다면 CPU로 러닝

    img_E = model(img_L)
    img_E = util.tensor2uint(img_E) #이미지(프레임) 형태로 변환
    img_E = cv2.cvtColor(img_E, cv2.COLOR_RGB2BGR)

    out.write(img_E)

    i += 1
    print('%d/%d'%(i, n_frames)) # 완료된 프레임 / 총 프레임

out.release()
cap.release()