import ultralytics
import torch
from collections import defaultdict
import numpy as np
import torch.backends.cudnn as cudnn
import yaml
from PIL import Image
from ultralytics import YOLO
import cv2



def load_config_data(data_path: str) -> dict: 
    """Функция для загрузки всех данных из конфигурационного файла yaml"""
    try:
        with open(data_path, 'r', encoding='utf8') as file:
            config_data = yaml.safe_load(file)
            return config_data
        
    except FileNotFoundError:
        print('Ошибка при загрузке конфигурационного файла')


# Загрузка всех переменных из конфигурационного файла
CONFIG_DATA = load_config_data('config.yaml')
print(CONFIG_DATA)



try:
    model = YOLO(CONFIG_DATA['yolov8_model'])
    model.to(device=torch.device("cuda"))
    print('Модель YOLO успешно загружена')
    print(f'Девайс для YOLO: {model.device}')
    
except Exception:
    print('Возникла ошибка при загрузке модели')




