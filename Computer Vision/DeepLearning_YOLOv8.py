# -*- coding: utf-8 -*-
# 코드 돌리기 전 data.yaml 생성하기

"""
# Ultralytics 및 기타 필요한 패키지 설치
!pip install ultralytics
!pip install requests
"""

import os
import zipfile
import requests
import ultralytics
import torch

import os
import glob


def main():
    # 환경 체크
    ultralytics.checks()

    from ultralytics import YOLO

    # GPU 사용 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # YOLOv8 모델 불러오기
    model = YOLO('yolov8s-seg.pt')  # YOLOv8의 사전 학습된 가중치 로드
    model = model.to(device)

    print(type(model.names), len(model.names))

    print(model.names)

    # 데이터 경로 및 기타 설정
    dataset_path = 'E:\\DatasetAssemble\\Dataset'
    data_config = os.path.join(dataset_path, 'data.yaml')  # yaml 파일 확인 설정

    # 초기 모델 학습 (조기 종료를 사용하여 에포크 수 설정)
    model.train(data=data_config,
                epochs=100,
                patience=10,
                batch=64,
                imgsz=416,
                save_dir='E:\\DatasetAssemble\\Dataset')

    # 단일 디렉토리 경로에 대한 test진행 코드
    #results = model.predict(source='E:\\DatasetAssemble\\Dataset\\Test\\images\\*.jpg', save=True) 

'''
    #피라미드 구조의 test 데이터셋에 대한 test 진행 코드
    # Set the base directory
    base_dir = r"E:\\DatasetAssemble\\Dataset\\Test\\TestImages"

    # Find all .jpg files in the base directory and its subdirectories
    jpg_files = []
    for root, dirs, files in os.walk(base_dir):
        jpg_files.extend(glob.glob(os.path.join(root, '*.jpg')))

    # Assuming model.predict can accept a list of file paths
    results = model.predict(source=jpg_files, save=True)
'''

if __name__ == '__main__':
    main()

""" 
여러번 학습해야 할 경우 사용-> 새로운 .py 만들어서 돌릴 것!
# 저장된 모델 로드
model = YOLO('path/to/initial_model/best.pt')  # best.pt는 학습된 모델의 최종 가중치 파일

# 추가 학습
model.train(data=data_config, epochs=50, patience=10, batch=64, imgsz=416, save_dir='path/to/updated_model')
"""
