import cv2 as cv
import numpy as np

#画像データの読み込み
img = cv.imread("a_4.jpg")

#BGR色空間からHSV色空間への変換
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

#色検出しきい値の設定
lower = np.array([0,0,100])
upper = np.array([180,45,255])

#色検出しきい値範囲内の色を抽出するマスクを作成
frame_mask = cv.inRange(hsv, lower, upper)

#論理演算で色検出
dst = cv.bitwise_and(img, img, mask=frame_mask)

cv.imshow("img", dst)

if cv.waitKey(0) & 0xFF == ord('q'):
    cv.destroyAllWindows()