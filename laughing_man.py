import io
import picamera
import cv2
import numpy as np
import math

# # カスケード分類器取得
cascade_path = "/home/pi/opencv/data/haarcascades/haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cascade_path)

stream = io.BytesIO()

color = (255,255,255)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

camera = picamera.PiCamera()
camera.resolution = (CAMERA_WIDTH, CAMERA_HEIGHT)

laugh = cv2.imread('/home/pi/laghfing_man/warai_flat.png', -1)  # 画像の読み込み。-1でアルファチャンネルを付けて読む
mask = cv2.cvtColor(laugh[:,:,3], cv2.COLOR_GRAY2BGR)/255.0  # 画像からアルファチャンネルだけを抜き出して0から1までの値にする。あと3チャンネルにしておく
laugh = laugh[:,:,:3]  # アルファチャンネルを消したもの

while True:
    camera.capture(stream, format='jpeg')
    data = np.fromstring(stream.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(data, 1)
    cv2.imshow('image',image)
    cv2.waitKey(16)

    # グレースケール変換
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    shape = math.floor(image.shape[1]/4)   # 小数点以下を切り捨てて整数に
    shape2 = math.floor(image.shape[0]/4)　 # 小数点以下を切り捨てて整数に
    image_gray = cv2.resize(image_gray, (shape, shape2))  # そのままだと遅かったので画像を4分の1にしてさらに高速化。

    facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))

    image_output = image
    
    if len(facerect) > 0: 

        # 検出した顔を囲む矩形の作成
        for rect in facerect:
            rect *= 4  # 認識を4分の1のサイズの画像で行ったので、結果は4倍しないといけない。

            # 画像のサイズ調整
            rect[0] -= min(25, rect[0])
            rect[1] -= min(25, rect[1])
            rect[2] += min(50, image_output.shape[1]-(rect[0]+rect[2]))
            rect[3] += min(50, image_output.shape[0]-(rect[1]+rect[3]))

            # 笑い男とマスクを認識した顔と同じサイズにリサイズする。
            laugh2 = cv2.resize(laugh, tuple(rect[2:]))
            mask2 = cv2.resize(mask, tuple(rect[2:]))

            # 笑い男の合成。
            image_output[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] = laugh2[:,:] * mask2 + image_output[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]] * (1.0 - mask2)
                        
    cv2.imshow('image_out',image_output)
    cv2.waitKey(16)

    stream.seek(0)
    stream.truncate()

    if cv2.waitKey(1) > 0:
        break

cv2.destroyAllWindows()