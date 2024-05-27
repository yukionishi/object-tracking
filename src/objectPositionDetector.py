import cv2
from cv2 import aruco
import numpy as np

#var
width, height = (500,500) # 変形後画像サイズ

#section
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
img = cv2.imread('inu.jpg')
corners, ids, rejectedImgPoints = aruco.detectMarkers(img, dictionary) # 検出

#section
# 時計回りで左上から順にマーカーの「中心座標」を m に格納
m = np.empty((4,2))
corners2 = [np.empty((1,4,2))]*4
for i,c in zip(ids.ravel(), corners):
  corners2[i] = c.copy()
m[0] = corners2[0][0][2] #id=0(left top), position of 2(right bottom)
m[1] = corners2[1][0][3] #id=1(right top), position of 3(left bottom)
m[2] = corners2[2][0][0] #id=2(right bottom), position of 0(left top)
m[3] = corners2[3][0][1] #id=3(left bottom), position of 1(right top)

marker_coordinates = np.float32(m)
true_coordinates   = np.float32([[0,0],[width,0],[width,height],[0,height]])
trans_mat = cv2.getPerspectiveTransform(marker_coordinates,true_coordinates)
img_trans = cv2.warpPerspective(img,trans_mat,(width, height))
cv2_imwrite(img_trans)
