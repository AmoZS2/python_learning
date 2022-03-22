# -*- coding: utf-8 -*-
import numpy as np
import cv2

def lowpass_filter(src, a = 0.5):
    # 高速フーリエ変換(2次元)
    src = np.fft.fft2(src)
    
    # 画像サイズ
    h, w = src.shape

    # 画像の中心座標
    cy, cx =  int(h/2), int(w/2)
    
    # フィルタのサイズ(矩形の高さと幅)
    rh, rw = int(a*cy), int(a*cx)

    # 第1象限と第3象限、第1象限と第4象限を入れ替え
    fsrc =  np.fft.fftshift(src)  

    # 入力画像と同じサイズで値0の配列を生成
    fdst = np.zeros(src.shape, dtype=complex)

    # 中心部分の値だけ代入（中心部分以外は0のまま）
    fdst[cy-rh:cy+rh, cx-rw:cx+rw] = fsrc[cy-rh:cy+rh, cx-rw:cx+rw]
    
    # 第1象限と第3象限、第1象限と第4象限を入れ替え(元に戻す)
    fdst =  np.fft.fftshift(fdst)

    # 高速逆フーリエ変換 
    dst = np.fft.ifft2(fdst)

    # 実部の値のみを取り出し、符号なし整数型に変換して返す
    return  np.uint8(dst.real)
    
def highpass_filter(src, a = 0.5):
    # 高速フーリエ変換(2次元)
    src = np.fft.fft2(src)
    
    # 画像サイズ
    h, w = src.shape

    # 画像の中心座標
    cy, cx =  int(h/2), int(w/2)
    
    # フィルタのサイズ(矩形の高さと幅)
    rh, rw = int(a*cy), int(a*cx)

    # 第1象限と第3象限、第1象限と第4象限を入れ替え
    fsrc =  np.fft.fftshift(src)  

    # 入力画像と同じサイズで値0の配列を生成
    fdst = fsrc.copy()

    # 中心部分だけ0を代入（中心部分以外は元のまま）
    fdst[cy-rh:cy+rh, cx-rw:cx+rw] = 0
    
    # 第1象限と第3象限、第1象限と第4象限を入れ替え(元に戻す)
    fdst =  np.fft.fftshift(fdst)

    # 高速逆フーリエ変換 
    dst = np.fft.ifft2(fdst)

    # 実部の値のみを取り出し、符号なし整数型に変換して返す
    return  np.uint8(dst.real)

def main():
    # 入力画像を読み込み
    img = cv2.imread("a_4.jpg")

    # グレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # ローパスフィルタ処理
    limg = lowpass_filter(gray, 0.3)

    # ハイパスフィルタ処理
    himg = highpass_filter(gray, 0.8)

    lframe = cv2.adaptiveThreshold(
        limg,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    hframe = cv2.adaptiveThreshold(
        himg,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # 処理結果を出力
    cv2.imshow("outl.png", limg)
    cv2.imshow("outh.png", himg)
    key = cv2.waitKey(0)

if __name__ == "__main__":
    main()