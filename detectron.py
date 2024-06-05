import ultralytics
import torch
import cv2
import warnings
import sys
import numpy as np

from pathlib import Path
from dataclasses import dataclass 
from time import time
from collections import defaultdict
from ultralytics import YOLO, RTDETR


sys.path.append("./")
from model.display import Display
from utils.paths import latest_video_file_num
from utils.video_utils import get_frame_size
from utils.logs import logger

warnings.filterwarnings('ignore')


@dataclass
class ModelDetectronConfig:
    #-------------------------------------Общие настройки-------------------------------------
    verbose: bool = True
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_type: ultralytics = YOLO
    model_path: str = r"./weights_base/yolov8n.pt"
    # model_path: str = r"D:\runs\tracker_vt_visdrone\weights\best.pt"

    save_video: bool = True 
    saved_video_folder: str = r"output_videos"
    saved_video_name: str = r"output_video" 
    saved_video_format: str = r".mp4"
    #--------------------------------------Режим трекинга-------------------------------------
    tracker: str =  r'botsort.yaml' 
    persist: bool = True
    #-----------------------------------------------------------------------------------------


 
class ModelDetectron(ModelDetectronConfig):
    """
    Класс для задачи детекции/трекинга
    на видеопотоке в real-time
      """
    
    def __init__(self):
        self.model = self.load_model()


    def load_model(self) -> RTDETR | YOLO:
        model = self.model_type(self.model_path)
        model = model.to(self.device)
        model.fuse()
        return model
    

    def video_saver(self) -> None:
        """
        Функция для сохранения видеопотока как в режиме трекинга,
        так и в режиме детекции в production
        """
        
        if self.save_video:
            Path(self.saved_video_folder).mkdir(parents=True, exist_ok=True)
            # Находим число сохранённых видео
            video_num = latest_video_file_num(ModelDetectronConfig())
            # Для сохранения модели в директорию output_videos
            self.save_path = f'{self.saved_video_folder}/{self.saved_video_name}{video_num}{self.saved_video_format}'
            save_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            save_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # С какими параметрами сохраняем видеопоток
            self.save_out = cv2.VideoWriter(f'{self.save_path}', save_fourcc, save_fps, (self.frame_width, self.frame_height))


    

    def stream_detection(self, input_video_path: str):
        """
        Обработка видеопотока в режиме детекции и без доп. функционала
        для внедрения в локальный сервер (production).
        Рекомендуется запускать на CUDA
        """

        self.cap = cv2.VideoCapture(input_video_path)

        self.frame_width, self.frame_height = get_frame_size(self.cap)

        assert self.cap.isOpened(), 'Ошибка при открытии видео-файла'

        self.video_saver()  # Настройка cохранения видеопотока

        while self.cap.isOpened():
            # Считывание фрейма из видео
            success, frame = self.cap.read()
    
            if not success:
                break
    
            else:
                # Запуск трекера
                results = self.model(frame)
    
                # Visualize the results on the frame
                frame = results[0].plot()


            if self.save_video:
                self.save_out.write(frame)
        

            # Остановка цикла при нажатии "q"
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
    
            yield(b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        
        # Сохранение видеопотока в указанную папку
        self.cap.release()

        if self.save_video:
            self.save_out.release()
            logger.info(f"Сохранённый видеопоток: {self.save_path}")

        cv2.destroyAllWindows()
        

    def stream_tracking(self, input_video_path: str) -> None:
        """
        Обработка видеопотока в режиме трекинга с доп функционалом.
        В production эта функция не пойдёт, она нужна для локального
        запуска для тестирования доп функционала модели.
        """

        self.cap = cv2.VideoCapture(input_video_path)

        self.frame_width, self.frame_height = get_frame_size(self.cap)

        assert self.cap.isOpened(), 'Ошибка при открытии видео-файла'

        self.video_saver()  # Настройка cохранения видеопотока

        # Отображение доп. информации на экран
        cls_display = Display(self.frame_width, self.frame_height) 

        # История линии трека
        track_history = defaultdict(lambda: [])

        # Цикл по видео фрейму
        while self.cap.isOpened():
            cls_display.start_time = time()
            # Считывание фрейма из видео
            success, frame = self.cap.read()            
            if not success:
                break

            # Запуск модели 
            results = self.model.track(frame,
                                       persist=self.persist,
                                       tracker=self.tracker,
                                       verbose=self.verbose)


            boxes = results[0].boxes.cpu()
            obj_count = len(boxes)  # Кол-во распознанных объектов

            if len(boxes) != 0: 
                # Выход модели для данного frame
                xyxys = list(boxes.xyxy.numpy().astype(int))
                ids = list(boxes.id.numpy().astype(int))      
                cls = list(boxes.cls.numpy().astype(int))
                conf = list(boxes.conf.numpy().astype(float))

                for idx in range(len(xyxys)):
                    cls_display.draw_boxing(frame, idx, xyxys, ids, cls, conf)

                    track = track_history[ids[idx]]
                    track.append((float((xyxys[idx][0] + xyxys[idx][2]) // 2), float((xyxys[idx][1]) + xyxys[idx][3]) // 2)) 
                    if len(track) > 60:  
                        track.pop(0)

                    # Отображение линии трека
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 0), thickness=2)
            
            # Вывод информации на дисплей
            cls_display.working_time_display(frame)
            cls_display.fps_display(frame)
            cls_display.project_name_display(frame)  
            cls_display.object_count_display(frame, obj_count)

            # Вывод кадра на экран
            cv2.imshow('Detectron: track mode', frame)

            if self.save_video:
                self.save_out.write(frame)
        
            # Остановка цикла при нажатии "q"
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        
        # Сохранение видеопотока в указанную папку
        self.cap.release()

        if self.save_video:
            self.save_out.release()
            logger.info(f"Сохранённый видеопоток: {self.save_path}")

        cv2.destroyAllWindows()



    def __call__(self, input_path: str, mode: str):
        pass 
        