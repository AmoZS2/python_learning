# import 
import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IEPlugin
 
# ターゲットデバイスの指定 
plugin = IEPlugin(device='MYRIAD')
 
# モデルの読み込みと入出力データのキー取得（顔検出） 
net_face  = IENetwork(model='intel/face-detection-retail-0005/FP16/face-detection-retail-0005.xml', weights='intel/face-detection-retail-0005/FP16/face-detection-retail-0005.bin')
exec_net_face  = plugin.load(network=net_face)
input_blob_face = next(iter(net_face.inputs))
out_blob_face  = next(iter(net_face.outputs))
 
# モデルの読み込みと入出力データのキー取得（landmarks） 
net_landmarks = IENetwork(model='intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml', weights='intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.bin')
exec_net_landmarks = plugin.load(network=net_landmarks)
input_blob_landmarks = next(iter(net_landmarks.inputs))
out_blob_landmarks = next(iter(net_landmarks.outputs))
 
# カメラ準備 
cap = cv2.VideoCapture(0)
 
#================================================== 
# メインループ 
#================================================== 
while True:
    # キー押下で終了 
    key = cv2.waitKey(1)
    if key != -1:
        break
 
    # カメラ画像読み込み 
    ret, frame = cap.read()
 
    # 入力データフォーマットへ変換 
    img = cv2.resize(frame, (300, 300)) # HeightとWidth変更 
    img = img.transpose((2, 0, 1))      # HWC > CHW 
    img = np.expand_dims(img, axis=0)   # CHW > BCHW 
 
    # 推論実行 
    out = exec_net_face.infer(inputs={input_blob_face: img})
 
    # 出力から必要なデータのみ取り出し 
    out = out[out_blob_face]
 
    # 不要な次元を削減 
    out = np.squeeze(out)
 
    # 検出されたすべての顔領域に対して１つずつ処理 
    for detection in out:
        # conf値の取得 
        confidence = float(detection[2])
 
        # バウンディングボックス座標を入力画像のスケールに変換 
        xmin = int(detection[3] * frame.shape[1])
        ymin = int(detection[4] * frame.shape[0])
        xmax = int(detection[5] * frame.shape[1])
        ymax = int(detection[6] * frame.shape[0])
 
        # conf値が0.5より大きい場合のみLandmarks推論とバウンディングボックス表示 
        if confidence > 0.5:
           # 顔検出領域はカメラ範囲内に補正する。特にminは補正しないとエラーになる 
            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > frame.shape[1]:
                xmax = frame.shape[1]
            if ymax > frame.shape[0]:
                ymax = frame.shape[0]
 
            #-------------------------------------------------- 
            #  ディープラーニングLandmarks推定 
            #-------------------------------------------------- 
            # 顔領域のみ切り出し 
            img_face = frame[ ymin:ymax, xmin:xmax ]
 
            # 入力データフォーマットへ変換 
            img = cv2.resize(img_face, (48, 48)) # HeightとWidth変更 
            img = img.transpose((2, 0, 1))       # HWC > CHW 
            img = np.expand_dims(img, axis=0)    # CHW > BCHW 
 
            # 推論実行 
            out = exec_net_landmarks.infer(inputs={input_blob_landmarks: img})
 
            # 出力から必要なデータのみ取り出し 
            out = out[out_blob_landmarks]
 
            # 不要な次元を削減 
            out = np.squeeze(out)
 
            # Landmarks検出位置にcircle表示 
            for i in range(0, 10, 2):
                x = int(out[i] * img_face.shape[1]) + xmin
                y = int(out[i+1] * img_face.shape[0]) + ymin
                cv2.circle(frame, (x, y), 10, (89, 199, 243), thickness=-1)
 
            # バウンディングボックス表示 
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(89, 199, 243), thickness=3)
 
    # 画像表示 
    cv2.imshow('frame', frame)
 
#================================================== 
# 終了処理 
#================================================== 
cap.release()
cv2.destroyAllWindows()