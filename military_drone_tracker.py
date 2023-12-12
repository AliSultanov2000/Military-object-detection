import yaml
from torch.cuda import is_available
from time import time 

from military_drone_display import Display
from ultralytics import YOLO
import cv2


# Важная ссылка
#  YOUTUBE: Live tracking on real FPV drone video | Autonomous Drone Object Tracking OpenCV Python   - отображение указателя 


class YoloModel:
    """Базовый класс для работы с YOLOv8"""

    def __init__(self, model_path):

        self.device = 'cuda' if is_available() else 'cpu'
        self.model_path = model_path
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.names
        self.start_time = 0
        self.end_time = 0
    
    
    def load_model(self):
        """Загрузка модели YOLOv8"""

        model = YOLO(self.model_path)
        model = model.to(device='cpu')
        model.fuse()
        return model


    def train_model(self, data, imgsz, epochs, batch, name):
        """Тренировка модели YOLOv8 для задачи трекинга"""
        self.model.train(
        data=data,  # Путь до YAML файла
        name=name,  # Название модели, которая сохранится 
        imgsz=imgsz,  
        epochs=epochs,  
        batch=batch)
    

    def test_model(self):
        """Запуск на тестирование модели YOLOv8"""
        # self.model.val()


# # Перейти полностью на OpenCV. Избавиться от двойного цикла. Протестировать на других видосах
#     def _box_plotting(self, results, frame): 
#         """Отображение на экране box-а c распознанными объектами"""
#         # for result in results: 
#         boxes = results[0].boxes.cpu().numpy()

#         xyxys = boxes.xyxy
#         for xyxy in xyxys:
#             cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
#             cv2.putText(frame, 'Танк', (int(xyxy[0]), int(xyxy[1]) - 4), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
#             # cv2.putText(frame, f'{confidence}', (int(xyxy[0] + 4), int(xyxy[1]) - 4), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
#         return frame
    

    def _box_plotting(self, results, frame):
        boxes = results[0].boxes
        if len(boxes) != 0:
            for i in range(len(results[0])):
                box = boxes[i]  # returns one box
                
                clsID = box.cls.numpy()[0]
                conf = box.conf.numpy()[0]
                xyxy = box.xyxy.numpy()[0]
                
    
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(frame, self.model.names[clsID], (int(xyxy[0]), int(xyxy[1]) - 4), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, str(round(conf, 3)), (int(xyxy[0]) + 90, int(xyxy[1]) - 4), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        return frame
