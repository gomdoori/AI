# 추가 학습하는 코드

# 경로 바꿨는지 체크!
# 코드 돌리기 전 data.yaml 생성하기

import os
import zipfile
import requests
import ultralytics

# 환경 체크
ultralytics.checks()

from ultralytics import YOLO

# 저장된 모델 로드
model = YOLO('path/to/initial_model/best-seg.pt')  # best.pt는 학습된 모델의 최종 가중치 파일

print(type(model.names), len(model.names))

print(model.names)

# 데이터 경로 및 기타 설정
dataset_path = 'path/to/dataset' #경로 설정
data_config = os.path.join(dataset_path, 'data.yaml') #야몰 설정

# 추가 학습
model.train(data=data_config,
            epochs=50,
            patience=10,
            batch=64,
            imgsz=416,
            save_dir='path/to/updated_model' # 학습된 YOLOv8 모델 저장 경로
            )
