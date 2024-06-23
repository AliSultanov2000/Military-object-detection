import torch
import cv2
import warnings
import sys
import numpy as np

from pathlib import Path
from dataclasses import dataclass, field
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
    clsses_weapon: tuple = (0, 2)
    person_name: str = "человек"
    armed_person_name: str = "вооруженный человек"

    colors: dict = field(default_factory=lambda: {
        ModelDetectronConfig.person_name: (255, 0, 0),
        ModelDetectronConfig.armed_person_name: (0, 0, 255)
        })
    
    save_video: bool = True 
    saved_video_folder: str = r"videos_output"
    saved_video_name: str = r"video_output" 
    saved_video_format: str = r".mp4"
    #-----------------------------------------------------------------------------------------


class ModelDetectron(ModelDetectronConfig):
    def __init__(self, model_type: str, model_weights: str):
        self.model = self.load_model(model_type, model_weights)

    def load_model(self, model_type: str, model_weights: str) -> YOLO | RTDETR:
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
    

    def check_bboxs_intersection(self, xyxys, cls, person_idx: int) -> bool:
        """
        Функция для проверки пересечения bounding boxes
        человека и оружия с целью выдать метку вооруженный человек 
        """
        for i in range(len(xyxys)):
            if cls[i] in self.clsses_weapon:
                # Проверяем, пересекаются ли bb person_idx и оружия
                box1 = xyxys[i]             # bbox оружия
                box2 = xyxys[person_idx]    # bbox человека
                b1_x1, b1_y1, b1_x2, b1_y2 = box1
                b2_x1, b2_y1, b2_x2, b2_y2 = box2
                # Зона пересечения
                inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * (
                    np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)
                ).clip(0)

                if inter_area != 0:
                    return True
        return False 
 
        
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
            # Считывание фрейма из видео
            success, frame = cap.read()            
            if not success:
                break

            # Запуск на детекцию
            #-------------------------------
            results = self.model(frame)
            boxes = results[0].boxes.cpu()
            #-------------------------------
            if len(boxes) != 0: 
                # Выход модели для данного frame
                #----------------------------------------------
                xyxys = boxes.xyxy.cpu().numpy().astype(int)
                cls = boxes.cls.cpu().numpy().astype(int)
                conf = boxes.conf.cpu().numpy().astype(float)
                #----------------------------------------------
                for idx in range(len(xyxys)):
                    if cls[idx] == 1:  # Если это человек
                        if self.check_bboxs_intersection(xyxys, cls, idx):
                            #------для доп.визуализации вооруж.человека------
                            cls_display.draw_ellipse(frame,
                                                     xyxys[idx])
                            #------------------------------------------------
                            person_type = self.armed_person_name
                        else:  # если нет пересечения
                            person_type = self.person_name
                        # Выводим на экран только человека
                        cls_display.draw_boxing(frame,
                                                xyxys[idx],
                                                f'{person_type} {conf[idx]:.2f}',
                                                ModelDetectronConfig().colors[person_type])
                        
            if self.save_video:
                save_out.write(frame)
        
            ret, buffer = cv2.imencode('.jpg', frame)

            frame = buffer.tobytes()

            # Передаём результат в сервер
            yield(b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cap.release()
        
        # Сохранение видеопотока в указанную папку
        if self.save_video:
            save_out.release()
            logger.info(f"Сохранённый видеопоток: {save_path}")

        cv2.destroyAllWindows()
