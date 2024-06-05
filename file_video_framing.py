import cv2

import time
from pathlib import Path


class DatasetVideoFraming:

    def __init__(self):
        self.interval = 20   # Интервал для нарезки видео
        self.img_index = 0   # Индекс сохранённого изображения

        self.outpul_subfolder_name = r"video_aleksey"          # Подпапка
        self.img_name = r'img'     # Базовое название сохранённых изображений
        self.img_format = r'.PNG'  # Формат сохранения изображений



    def _video_framing(self, video_input_path: str, video_output_path: str) -> None:
        """
           Обработка и сохранение одного видеопотока
           video_input_path: абсолютный путь до видеопотока
           video_output_path: абсолютный путь до папки куда сохраняем кадры
        """

        cap = cv2.VideoCapture(video_input_path)
        assert cap.isOpened(), 'Ошибка при открытии видео-файла'
        index = 0
        while cap.isOpened(): 
            ret, frame = cap.read()
            if not ret:
                break
    
            if index % self.interval == 0:
                cv2.imwrite(f"{video_output_path}\{self.img_name}{self.img_index}{self.img_format}", frame)
                self.img_index += 1

            index += 1

    
    def all_videos_framing(self, input_folder_path: str, output_folder_path: str):
        """
            Обработка и сохранение всех видео-файлов из папки
            input_folder_path: путь до папки с видео
            output_folder_path: путь до папки куда сохраняем
        """

        start_time = time.time() 
        video_paths = list(Path(input_folder_path).iterdir())  # Абсолютные пути до всех видеопотоков
        video_nums = len(video_paths)                               # Кол-во видео на обработку
        print(f'Общее кол-во видео на обработку: {video_nums}')
        
        for i in range(len(video_paths)):
            # Обрабатываем одно видео
            input_video_path = str(video_paths[i])
            print(f'Идёт обработка видео: {input_video_path}')
            # Создаём папку для сохранения видео
            output_video_path = f'{output_folder_path}\{self.outpul_subfolder_name}{str(i)}'

            if Path.exists(output_video_path) == False:
                Path(output_video_path).mkdir(parents=True, exist_ok=True)
                # Запускаем фрейминг, сохраняем в эту папку все изображения
                self._video_framing(input_video_path, output_video_path)
        
        print('=' * 150)
        print(f'Общее кол-во полученных изображений: {self.img_index + 1}')
        print(f'Общее время обработки всех видеопотоков: {time.time() - start_time} сек.')




if __name__ == "__main__":
    framer = DatasetVideoFraming()
    framer._video_framing(r"C:\Users\1NR_Operator_33\Desktop\Трекер датасет\video_2024-05-22_19-26-32.mp4", r"D:/rayevskiy_dataset/ot_alekseya")