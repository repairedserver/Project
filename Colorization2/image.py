import torch #pytorch
import cv2 #opencv-python
import os #운영체제 제공 기능 모듈
from utils import utils_image as util #유틸 모듈
from models.network_rrdbnet import RRDBNet as net #models 폴더에서 RRDBNet 임포트

#데이터셋
#https://github.com/cszn/KAIR/releases/download/v1.0/BSRGAN

img_path = 'images/iu.jpg' #화질을 향상시킬 이미지의 경로

#학습시킬 프로세서 cuda:GPU, cpu:CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드
model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4)
model.load_state_dict(torch.load(os.path.join('model', 'BSRGAN.pth')), strict=True)
model = model.to(device)
model.eval() 

#이미지 로드
with torch.no_grad():
    img = cv2.imread(img_path)

    img_L = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB) #4채널, png 이미지라면 3채널로 변경
    img_L = util.uint2tensor4(img_L) #텐서 형태로 변환
    img_L = img_L.to(device) #GPU가 있다면 GPU로, 없다면 CPU로 러닝

    img_E = model(img_L)

    img_E = util.tensor2uint(img_E) #이미지 형태로 변환
    util.imsave(img_E, os.path.splitext(img_path)[0] + '_adv.jpg') #이름을 {name}_adv.jpg 형태로 저장