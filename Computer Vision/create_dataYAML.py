# 데이터 경로 정확한지 확인

import yaml

"""
            0: "흑색연기",
            1: "백색/회색연기",
            2: "화염",
            3: "구름",
            4: "안개/연무",
            5: "굴뚝연기",
           
"""

def create_data_yaml(train_path, val_path, output_path, test_path=None):
    data = {
        'train': train_path,
        'val': val_path,
        'nc': 6,
        'names': {
            0: '흑색연기',
            1: '백색/회색연기',
            2: '화염',
            3: '구름',
            4: '안개/연무',
            5: '굴뚝연기'
        }
    }

    if test_path:
        data['test'] = test_path

    with open(output_path, 'w') as outfile:
        yaml.dump(data, outfile, allow_unicode=True, default_flow_style=False)  # 읽기 쉬운 블록 스타일로 출력

    print("dataset.yaml 파일이 생성되었습니다.")


train_path = 'E:\\DatasetAssemble\\Dataset\\Train'  # 훈련 데이터 경로
val_path = 'E:\\DatasetAssemble\\Dataset\\Valid'  # 검증 데이터 경로

output_path = 'E:\\DatasetAssemble\\Dataset\\data.yaml'  # 생성될 data.yaml 파일 경로


# 테스트(test) 데이터셋이 없는 경우
create_data_yaml(train_path, val_path, output_path)


'''
# 테스트(test) 데이터셋이 있는 경우
test_path = 'E:\\DatasetAssemble\\Dataset\\Test'
create_data_yaml(train_path, val_path, class_list, output_path, test_path)
'''