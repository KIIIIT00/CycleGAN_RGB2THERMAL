import cv2
import os

# 入出力フォルダ
IMPORT_FILE = './results/RtoT300/test_latest/images/'
OUTPUT_FILE = './cascade_results/RtoT300/'
REAL_A = 'real_A.png'
REAL_B = 'real_B.png'
FAKE_B = 'fake_B.png'

faceCascade = cv2.CascadeClassifier('./val/haarcascade/haarcascade_fullbody.xml')

# INPUT_FILEを走査
for filename in os.listdir(IMPORT_FILE):
    # 特定の名前がファイル名に含まれるかをチェック
    if REAL_A in filename:
        print(filename)
        # 画像読み込み
        img = cv2.imread(os.path.join(IMPORT_FILE, filename))
        # 物体認識（人）の実行
        facerect = faceCascade.detectMultiScale(img, scaleFactor=1.01, minNeighbors=1, minSize=(10, 10))

        cv2.imwrite(os.path.join(OUTPUT_FILE, filename), img)
    
    if REAL_B in filename:
        real_b = cv2.imread(os.path.join(IMPORT_FILE, filename))
        cv2.imwrite(os.path.join(OUTPUT_FILE, filename), real_b)
    
    if FAKE_B in filename:
        fake_b = cv2.imread(os.path.join(IMPORT_FILE, filename))
        cv2.imwrite(os.path.join(OUTPUT_FILE, filename), fake_b)
