from torch.cuda import is_available
from time import time 
from drone_display import Display
from ultralytics import YOLO
import cv2

import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)



class YoloModel:
    """Базовый класс для работы с YOLOv8"""

    def __init__(self, model_path):

        self.device = 'cuda' if is_available() else 'cpu'
        self.model_path = model_path
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.names

        
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


    def _box_plotting(self, results):
        """Отображение на экране box-а c распознанными объектами"""
    
        # Получение всех боксов
        boxes = results[0].boxes.xywh.cpu()
        # Визуализация результата на фрейме
        annotated_frame = results[0].plot()
        # Отрисовка бокса с указанием id трека 
        for box in boxes:
            x, y, w, h = box
            x, y, w, h = int(x), int(y), int(w), int(h)
            # Отрисовка центров для Bounding Boxes
            cv2.drawMarker(annotated_frame, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)

        return annotated_frame 
    


class YoloTracker(YoloModel):
    """Класс YOLOv8 для задачи трекинга объектов"""

    def __init__(self, model_path):
        super().__init__(model_path)


    def image_tracking(self, image_path: str) -> None: 
        """Предсказание модели на изображении"""
        # Список с результатами
        results = self.model.track(image_path, persist=True)
        # Отображение на экран
        annotated_frame = self._box_plotting(results)
        cv2.imshow("Sharp eye", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
 
        
    def stream_tracking(self, video_path: str, cls_display: Display) -> None: 
        """Предсказание модели на видеопотоке"""

        # Открытие видео файла 
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        assert cap.isOpened()                          
        
        # Цикл по видео фрейму
        while cap.isOpened():
            cls_display.start_time = time()
            # Считывание фрейма из видео
            success, frame = cap.read()

            if success:
                # Запуск YOLOv8 трекера
                results = self.model.track(frame, persist=True)
                # Отображение на экране box-ов
                annotated_frame = self._box_plotting(results) 

                # Вывод времени работы, fps, эмблемы на дисплей
                cls_display.working_time_display(annotated_frame)
                cls_display.fps_display(annotated_frame)
                cls_display.logotype_display(annotated_frame)
                cls_display.project_name_display(annotated_frame, 640, 360)
                cls_display.target_aiming_display(annotated_frame, 640, 360)  # Координаты x_center, y_center

        

                # Вывод кадра на экран
                cv2.imshow('YOLOv8 tracker', annotated_frame)
                frame_count += 1
                # Остановка цикла при нажатии "q"
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Остановка цикла, если видео закончено
                break
        
        # Закрытие видеопотока
        cap.release()
        cv2.destroyAllWindows()
  




if __name__ == '__main__':
    # Cоздание экземпляра класса
    tracker = YoloTracker(r"C:\Users\1NR_Operator_33\runs\detect\yolov8_tank_detection42\weights\best.pt")
    logging.info(f'Загружена модель: {tracker.model_path}')
    logging.info(f'Используемый девайс: {tracker.device}')
