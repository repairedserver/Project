import numpy as np
import cv2

print("모델을 불러오는 중...")
net = cv2.dnn.readNetFromCaffe('./model/colorization_deploy_v2.prototxt', './model/colorization_release_v2.caffemodel')
pts = np.load('./model/pts_in_hull.npy')


class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)

net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype='float32')]


image = cv2.imread('./images/1_result.png') #여기에 이미지 파일명 입력
scaled = image.astype("float32") / 255.0
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)


resized = cv2.resize(lab, (224, 224))
L = cv2.split(resized)[0]
L -= 50


net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)

colorized = (255 * colorized).astype("uint8")

resize_org = cv2.resize(image, (720, 540))
resize_color = cv2.resize(colorized, (720, 540))

cv2.imshow("Original", resize_org)
cv2.imshow("Colorized`", resize_color)

cv2.waitKey(0)
