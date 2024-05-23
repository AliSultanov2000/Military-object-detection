import cv2

import time
import argparse
from dataclasses import dataclass
from pathlib import Path



class TrackerDataset:
    def __init__(self, input_folder_path: str, output_folder_path: str):
        self.input_folder_path = input_folder_path     # Путь до папки с видео

        self.output_folder_path = output_folder_path   # Путь до папки куда сохраняем
        self.outpul_subfolder_name = r"video"          # Подпапка


        self.interval = 10   # Интервал для нарезки видео
        self.img_index = 0   # Индекс сохранённого изображения

        self.img_name = r'img'     # Базовое название сохранённых изображений
        self.img_format = r'.PNG'  # Формат сохранения изображений


    def _video_framing(self, video_input_path: str, video_output_path: str) -> None:
        """Обработка и сохранение одного видеопотока"""

        cap = cv2.VideoCapture(video_input_path)
        assert cap.isOpened(), 'Ошибка при открытии видео-файла'
        index = 0
        while cap.isOpened(): 
            ret, frame = cap.read()
            if not ret:
                break
    
            if index % self.interval == 0:
                cv2.imwrite(f"{video_output_path}/{self.img_name}{self.img_index}{self.img_format}", frame)
                self.img_index += 1

            index += 1

    
    def all_videos_framing(self):
        """Обработка и сохранение всех видеопотоков из папки"""
        start_time = time.time() 
        video_paths = list(Path(self.input_folder_path).iterdir())  # Абсолютные пути до всех видеопотоков
        video_nums = len(video_paths)                               # Кол-во видео на обработку
        print(f'Общее кол-во видео на обработку: {video_nums}')
        
        for i in range(len(video_paths)):
            # Обрабатываем одно видео
            input_video_path = str(video_paths[i])
            print(f'Идёт обработка видео: {input_video_path}')
            # Создаём папку для сохранения видео
            output_video_path = f'{self.output_folder_path}/{self.outpul_subfolder_name}{str(i)}'
            Path(output_video_path).mkdir(parents=True, exist_ok=True)
            # Запускаем фрейминг, сохраняем в эту папку все изображения
            self._video_framing(input_video_path, output_video_path)
        
        print(f'Общее время обработки всех видеопотоков: {time.time() - start_time} сек.')
        print(f'Общее кол-во полученных изображений: {self.img_index + 1}')


    # Добавление путей с argparse
    # Проверка существования папки
                




if __name__ == '__main__':
    dataset = TrackerDataset(r"C:\Users\1NR_Operator_33\Desktop\Трекер датасет", r"D:\rayevskiy_dataset")
    dataset.all_videos_framing()