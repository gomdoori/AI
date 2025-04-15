import os

cache_files = [
    "E:\\DatasetAssemble\\Dataset\\Train\\TrainImages\\NegativeDB_구름\\강원도강릉시강동면모전리_BC002001\\001_없음.cache",
    "E:\\DatasetAssemble\\Dataset\\Valid\\ValidImages\\NegativeDB_구름\\강원도강릉시강동면모전리_BC002001\\001_없음.cache"
]

for cache_file in cache_files:
    if os.path.exists(cache_file):
        os.remove(cache_file)
