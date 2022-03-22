import cv2 as cv
import numpy as np
from pyzbar.pyzbar import decode

#Webカメラの読み込み
image = cv.imread('a_4.jpg')

# 2値化の初期値
blocksize = 11
threshold = 2

# 画像をグレースケールに変換
gray_frame = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

# グレースケール用ノイズ除去
gray_frame = cv.fastNlMeansDenoising(gray_frame)

# 適応的閾値処理を用いて2値化
mono_frame = cv.adaptiveThreshold(
    gray_frame,
    255,
    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv.THRESH_BINARY,
    blocksize,
    threshold
)

# カラー用ノイズ除去
color_frame = cv.fastNlMeansDenoisingColored(image)
gray_frame2 = cv.cvtColor(color_frame, cv.COLOR_RGB2GRAY)

# 適応的閾値処理を用いて2値化
col_frame = cv.adaptiveThreshold(
    gray_frame2,
    255,
    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv.THRESH_BINARY,
    blocksize,
    threshold
)

#バーコードからデータを読み取る
for barcode in decode(mono_frame):
    #QRコードデータはバイトオブジェクトなので、カメラ上に描くために、文字列に変換する
    decoded = barcode.data.decode('utf-8')
    #QRコードの周りに長方形を描画しデータを表示する
    pts =np.array([
        barcode.polygon],
        np.int32
    )
    # QRの周りにポリゴンを描画
    cv.polylines(
        image,
        [pts],True,(255,0,0),
        5
    )
    pts2 = barcode.rect
    # 読取ったQRコードの内容を画面に表示
    cv.putText(
        image,
        decoded,
        (pts2[0] ,pts2[1]),
        cv.FONT_HERSHEY_COMPLEX,
        1,
        (0,0,0)
        ,2
    )

#バーコードからデータを読み取る
for barcode in decode(col_frame):
    #QRコードデータはバイトオブジェクトなので、カメラ上に描くために、文字列に変換する
    decoded = barcode.data.decode('utf-8')
    #QRコードの周りに長方形を描画しデータを表示する
    pts =np.array([
        barcode.polygon],
        np.int32
    )
    # QRの周りにポリゴンを描画
    cv.polylines(
        image,
        [pts],True,(0,255,0),
        5
    )
    pts2 = barcode.rect
    # 読取ったQRコードの内容を画面に表示
    cv.putText(
        image,
        decoded,
        (pts2[0] ,pts2[1]),
        cv.FONT_HERSHEY_COMPLEX,
        1,
        (0,0,0)
        ,2
    )

cv.imshow('col', col_frame)
cv.imshow('mono', mono_frame)
cv.imshow('real', image)

#qキーが押されるまで待機
#それぞれのキーで値を変更可能
key = cv.waitKey(0)