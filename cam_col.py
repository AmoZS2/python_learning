import cv2 as cv
import numpy as np
from pyzbar.pyzbar import decode

#Webカメラの読み込み
cap = cv.VideoCapture(1, cv.CAP_DSHOW)

#出力ウィンドウの設定
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

# 2値化の初期値
blocksize = 11
threshold = 2

# キャプチャしている間QR読取を続ける
while cap.isOpened():
    #カメラからフレーム情報を取得
    ret, frame = cap.read()

    # カラー用ノイズ除去
    color_frame = cv.fastNlMeansDenoisingColored(frame)
    gray_frame = cv.cvtColor(color_frame, cv.COLOR_RGB2GRAY)

    # 適応的閾値処理を用いて2値化
    col_frame = cv.adaptiveThreshold(
        gray_frame,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        blocksize,
        threshold
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
            frame,
            [pts],True,(0,255,0),
            5
        )
        pts2 = barcode.rect
        # 読取ったQRコードの内容を画面に表示
        cv.putText(
            frame,
            decoded,
            (pts2[0] ,pts2[1]),
            cv.FONT_HERSHEY_COMPLEX,
            1,
            (0,0,0)
            ,2
        )

    cv.imshow('bin', col_frame)
    cv.imshow('real', frame)

    #qキーが押されるまで待機
    #それぞれのキーで値を変更可能
    key = cv.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('a'):
        blocksize += 2
        print(f'blocksize: {blocksize}')
    elif key == ord('f'):
        blocksize -= 2
        print(f'blocksize: {blocksize}')
    elif key == ord('e'):
        threshold += 1
        print(f'Threshold: {threshold}')
    elif key == ord('w'):
        threshold -= 1
        print(f'Threshold: {threshold}')