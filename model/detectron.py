import torch
import cv2
import warnings
import sys

from pathlib import Path
from dataclasses import dataclass 
from time import time
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
    run_cuda = True
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() and run_cuda else 'cpu')

    show_classes: tuple = ()

    save_video: bool = True 
    saved_video_folder: str = r"output_videos"
    saved_video_name: str = r"output_video" 
    saved_video_format: str = r".mp4"
    #-----------------------------------------------------------------------------------------


class ModelDetectron(ModelDetectronConfig):
    def __init__(self, model_type: str, model_weights: str):
        self.model = self.load_model(model_type, model_weights)


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


    def __call__(self, input_video_path: str):
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

        # Отображение доп.информации на экран
        cls_display = Display(self.frame_width, self.frame_height)

        while cap.isOpened():
            cls_display.start_time = time()
            # Считывание фрейма из видео
            success, frame = cap.read()            
            if not success:
                break

            # Запуск на детекцию
            results = self.model(frame)
            boxes = results[0].boxes.cpu()
    
            if len(boxes) != 0: 
                # Выход модели для данного frame
                xyxys = list(boxes.xyxy.numpy().astype(int))
                cls = list(boxes.cls.numpy().astype(int))
                conf = list(boxes.conf.numpy().astype(float))
            
                for idx in range(len(xyxys)):
                    cls_display.draw_boxing(frame, idx, xyxys, cls, f"{self.model.names[cls[idx]]} {conf[idx]:.2f}")

            if self.save_video:
                save_out.write(frame)
        
            ret, buffer = cv2.imencode('.jpg', frame)

            frame = buffer.tobytes()
    
            yield(b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


        # Сохранение видеопотока в указанную папку
        cap.release()

        if self.save_video:
            save_out.release()
            logger.info(f"Сохранённый видеопоток: {save_path}")

        cv2.destroyAllWindows()
