"""
CycleGANの入力RGB画像に対して,yoloを用い,バウンディングボックスを描画する
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os

# yoloのモデルパス
yolo_model_path = "yolov8n.pt"

model = YOLO(yolo_model_path)

# 入出力フォルダ
IMPORT_FILE = './results/RtoT300/test_latest/images/'
OUTPUT_FILE = './cascade_results/RtoT300/'
REAL_A = 'real_A.png'
REAL_B = 'real_B.png'
FAKE_B = 'fake_B.png'

def people_detection(model, input_image_path, output_image_path):
    """
    人物の検出とバウンディングボックスの描画

    Parameters:
    model (YOLO) -- yoloのモデル
    input_image_path (str) -- 入力画像のパス
    output_image_path (str) -- 出力画像のパス

    Returns:
    boxes (numpy array) -- 検出された人物のバウンディングボックス
    names (numpy array) -- 検出された人物の名前
    """
    # 予測を実行
    results = model(input_image_path)

    # 結果を取得
    boxes = results.boxes
    names = results.names

    return boxes, names


def sort_by_images(image_path):
    """
    REAL_A,REAL_B,FAKE_Bにわける
    Parameters:
    image_path (str) -- 入力画像のパス

    Returns:
    realA_list (list) -- REAL_A画像のパス
    realB_list (list) -- REAL_B画像のパス
    fakeB_list (list) -- FAKE_B画像のパス
    """
    realA_list = []
    realB_list = []
    fakeB_list = []
    # 入力フォルダを走査
    for filename in os.listdir(image_path):
        if REAL_A in filename:
            realA_list.append(filename)
        elif REAL_B in filename:
            realB_list.append(filename)
        elif FAKE_B in filename:
            fakeB_list.append(filename)
    
    return realA_list, realB_list, fakeB_list


