import numpy as np
import cv2

print("모델을 불러오는 중...")
net = cv2.dnn.readNetFromCaffe('./model/colorization_deploy_v2.prototxt', './model/colorization_release_v2.caffemodel') #model 폴더에서 모델 불러오기
pts = np.load('./model/pts_in_hull.npy') #넘파이 배열 파일 로드


class8 = net.getLayerId("class8_ab") #레이어의 문자열 이름을 정수 식별자로 변환
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)

net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype='float32')]


image = cv2.imread('./images/iu_adv.jpg') #채색할 이미지 파일명 입력 ex) images/{name}_adv.jpg
scaled = image.astype("float32") / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB) 


resized = cv2.resize(lab, (224, 224)) # 이미지 비율 지정
L = cv2.split(resized)[0]
L -= 50


net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

ab = cv2.resize(ab, (image.shape[1], image.shape[0])) # 이미지 비율 지정

L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)

colorized = (255 * colorized).astype("uint8")

cv2.imwrite('images/iu_adv_color.jpg', colorized) #파일의 저장될 이름 입력 ex) image/{name}_adv_color.jpg

cv2.waitKey(0)
cv2.destroyAllWindows()