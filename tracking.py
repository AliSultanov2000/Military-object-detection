import yaml
import torch
from time import time   
import numpy as np
from PIL import Image
from ultralytics.utils.plotting import Annotator, colors
from ultralytics import YOLO
import cv2

# Важная ссылка
# https://docs.ultralytics.com/guides/security-alarm-system/#email-send-function



class YoloModel:
    """Базовый класс для работы с YOLOv8"""

    def __init__(self, model_path):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = model_path
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.start_time = 0
        self.end_time = 0

    
    def load_model(self):
        """Загрузка модели YOLOv8"""

        model = YOLO(self.model_path)
        model = model.to(device='cpu')
        model.fuse()
        return model


    def save_model(self):
        """Сохранение модели YOLOv8"""
        pass


    def test_model(self):
        """Запуск на тестирование модели"""
        pass


class Display:
    """Класс для отображения дополнительной информации о работе трекера на экран"""

    def fps_display(self, im0) -> None:
        """Отображение на экран FPS"""
        self.end_time = time()
        fps = 1 / np.round(self.end_time - self.start_time, 2)
        text = f'FPS: {int(fps)}'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        gap = 10

        cv2.rectangle(im0, (1130 - gap, 50 - text_size[1] - gap), (1130 + text_size[0] + gap, 50 + gap), (56, 84, 41), -1)
        cv2.putText(im0, text, (1130, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    

    def starting_time_display(self, im0) -> None: 
        """Отображение на экран времени работы"""
        pass

    def logotype_display(self, im0) -> None:
        """Отображение логотипа"""
        pass


    

# Перевести в отдельный проект
# Вынести отображение дисплея в отдельный класс ? + 
# Создать ipynb для EDA датасета
# ToDo сделать метод box_plot (как на изображении, так и на видео)
# Пофиксить костыль с try-except + 
# Пофиксить отображение FPS / Чекнуть правильность +
# Заполнить save_model, test_model
# Перевести все комментарии на русский язык +
# Перевести отображение класса на русский язык
# Добавить отображение времени работы трекера
# Разобраться с colors



class YoloTracker(YoloModel, Display):
    """Класс YOLOv8 для задачи трекинга объектов"""

    def __init__(self, model_path):
        super().__init__(model_path)


    def image_tracking(self, image_path: str) -> None: 
        """Предсказание модели на изображении"""

        results = self.model.track(image_path, persist=True)  # Cписок с результатами
        # Отображение результатов
        for result in results:
            im_array = result.plot()  # Список BGR в numpy array 
            im = Image.fromarray(im_array[..., ::-1])  # Переводим в RGB PIL image
            im.show()  # Отображение изображения

    
    def stream_tracking(self, video_path: str) -> None: 
        """Предсказание модели на видеопотоке"""

        # Открытие видео файла 
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frame_count = 0
        assert cap.isOpened()

        # Цикл по видео фрейму
        while cap.isOpened():
            self.start_time = time()
            # Считывание фрейма из видео
            success, frame = cap.read()

            if success:
                # Запуск YOLOv8 трекера
                results = self.model.track(frame, persist=True)
                # Получение всех боксов
                boxes = results[0].boxes.xywh.cpu()
        
                # Визуализация результата на фрейме
                annotated_frame = results[0].plot()
        
                # Отрисовка бокса с указанием id трека 
                for box in boxes:
                    x, y, w, h = box
                    x, y, w, h = int(x), int(y), int(w), int(h)
        
                    # Отрисовка центров для Bounding Boxes
                    cv2.circle(annotated_frame, (x, y), 5, (0, 0, 255), -1)
        
                # Вывод fps, времени работы, эмблемы на дисплей
                self.fps_display(annotated_frame)
                self.starting_time_display(annotated_frame)
                self.logotype_display(annotated_frame)

                # Вывод кадра на экран
                cv2.imshow("YOLOv8 Tracking", annotated_frame)
            
                frame_count += 1
                # Остановка цикла при нажатии "q"
                if cv2.waitKey(1) & 0xFF == ord("q"):
                     break
            else:
                # Остановка цикла если видео закончено
                break
        
        # Закрытие видео отображения
        cap.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    # Cоздание экземпляра класса
    tracker = YoloTracker(r"C:\Users\1NR_Operator_33\runs\detect\yolov8_tank_detection42\weights\best.pt")

    print(f'Загружена модель: {tracker.model_path}')
    print(f'Используемый девайс: {tracker.device}')

    # # Запуск на трекиг видеопотока
    tracker.stream_tracking(r"C:\Users\1NR_Operator_33\Downloads\танк Т90м Прорыв лучший танк в мире в бою.mp4")


