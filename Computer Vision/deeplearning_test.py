# -*- coding: utf-8 -*-
"""DeepLearning_Test.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1cRz1wCyK3iNdmqTHGXI5wqRTs0bwaHrJ
"""

# Ultralytics 및 기타 필요한 패키지 설치
!pip install ultralytics
!pip install requests

"""데이터 셋 url: https://universe.roboflow.com/ember-9pwq0/fire-tzv16

"""

import os
import zipfile
import requests

# RoboFlow 데이터셋 다운로드 링크
roboflow_url = "https://universe.roboflow.com/ds/ckncYYKVPg?key=CbSZ7rI1ky"

# 데이터셋 다운로드 경로
dataset_path = "/content/Dataset_for_yolov8"

# 데이터셋 압축 파일 경로
zip_path = os.path.join(dataset_path, "roboflow_dataset.zip")

# 데이터셋 다운로드
os.makedirs(dataset_path, exist_ok=True)
response = requests.get(roboflow_url)
with open(zip_path, 'wb') as f:
    f.write(response.content)

# 데이터셋 압축 해제
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(dataset_path)

# 압축 파일 삭제
os.remove(zip_path)

import ultralytics
# 환경 체크
ultralytics.checks()

from ultralytics import YOLO

# YOLOv8 모델 불러오기
model = YOLO('yolov8m.pt')

print(type(model.names), len(model.names))

print(model.names)

# 모델 학습
model.train(data=os.path.join(dataset_path, 'data.yaml'), epochs=10, patience=30, batch=32, imgsz=416)

results = model.predict(source='/content/Dataset_for_yolov8/test/images/*.jpg', save=True)