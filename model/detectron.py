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
from utils.video_utils import get_frame_size, bbox_center
from utils.logs import logger

warnings.filterwarnings('ignore')


@dataclass
class ModelDetectronConfig:
    #-------------------------------------Общие настройки-------------------------------------
    verbose: bool = True
    run_cuda = True
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() and run_cuda else 'cpu')

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
    
    def __init__(self, model_type: str, model_weights: str, mode: str):
        self.model = self.load_model(model_type, model_weights)
        self.mode = mode

        assert mode in ['detection', 'tracking'], 'Корректно введите режим запуска модели'
        
        if mode == 'tracking':
            self.track_history = defaultdict(lambda: [])
            self.base_func = self.tracking_mode
        else:
            self.base_func= self.detection_mode


    def load_model(self, model_type: str, model_weights: str) -> RTDETR | YOLO:
        """Возвращает модель: либо YOLO, либо RTDETR"""

        model_type = model_type.lower()
        assert model_type in ['yolo', 'rtdetr']

        if model_type == 'yolo':
            model = YOLO(model_weights)

        elif model_type == 'rtdetr':
            model = RTDETR(model_weights)
        
        model = model.to(self.device)
        model.fuse()
        return model
    


    def detection_mode(self, frame):
        """Forward pass модели в режиме детекции"""

        results = self.model(frame)
        boxes = results[0].boxes.cpu()

        if len(boxes) != 0: 
            # Выход модели для данного frame
            xyxys = list(boxes.xyxy.numpy().astype(int))
            cls = list(boxes.cls.numpy().astype(int))
            conf = list(boxes.conf.numpy().astype(float))
        
            for idx in range(len(xyxys)):
                self.cls_display.draw_boxing(frame, idx, xyxys, cls, f"{self.model.names[cls[idx]]} {conf[idx]:.2f}")
        
            

    def tracking_mode(self, frame):
        """Forward pass модели в режиме трекинга"""
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
                #-------------------------Отображение линии трека-------------------------
                center_coord = bbox_center(xyxys, idx)

                track = self.track_history[ids[idx]]
                track.append(center_coord) 
                if len(track) > 60:  
                    track.pop(0)
                
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))

                self.cls_display.draw_boxing(frame, idx, xyxys, cls, f"id:{ids[idx]} {self.model.names[cls[idx]]} {conf[idx]:.2f}")
                self.cls_display.draw_track_lines(frame, points, center_coord)
                #-------------------------------------------------------------------------
        
        # Вывод информации на дисплей
        self.cls_display.working_time_display(frame)
        self.cls_display.fps_display(frame)
        self.cls_display.project_name_display(frame)  
        self.cls_display.object_count_display(frame, obj_count)
    


    def __call__(self, input_video_path: str):
        """
        Обработка видеопотока в режиме детекции и без доп. функционала
        для внедрения в локальный сервер (production).
        Рекомендуется запускать на CUDA
        """

        cap = cv2.VideoCapture(input_video_path)

        self.frame_width, self.frame_height = get_frame_size(cap)

        assert cap.isOpened(), 'Ошибка при открытии видео-файла'

        # Сохранение видеопотока
        if self.save_video:
            Path(self.saved_video_folder).mkdir(parents=True, exist_ok=True)
            
            video_num = latest_video_file_num(ModelDetectronConfig())
            save_path = f'{self.saved_video_folder}/{self.saved_video_name}{video_num}{self.saved_video_format}'

            save_fps = int(cap.get(cv2.CAP_PROP_FPS))
            save_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            save_out = cv2.VideoWriter(f'{save_path}', save_fourcc, save_fps, (self.frame_width, self.frame_height))

        # Отображение доп. информации на экран
        self.cls_display = Display(self.frame_width, self.frame_height)

        while cap.isOpened():
            self.cls_display.start_time = time()
            # Считывание фрейма из видео
            success, frame = cap.read()            
            if not success:
                break

            # Запуск либо в режиме трекинга либо в режиме детекции в зав-ти от mode
            self.base_func(frame)

            if self.save_video:
                save_out.write(frame)
        
            ret, buffer = cv2.imencode('.jpg', frame)

            frame = buffer.tobytes()
    
            yield(b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            # # Вывод кадра на экран
            # cv2.imshow('detectron', frame)
        
            # # Остановка цикла при нажатии "q"
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break

                
        # Сохранение видеопотока в указанную папку
        cap.release()

        if self.save_video:
            save_out.release()
            logger.info(f"Сохранённый видеопоток: {save_path}")

        cv2.destroyAllWindows()
