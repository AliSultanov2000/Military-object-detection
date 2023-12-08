import yaml
from torch.cuda import is_available
from time import time   

from military_drone_display import Display
from ultralytics import YOLO
import cv2


class YoloModel:

    def __init__(self, model_path):

        self.device = 'cuda' if is_available() else 'cpu'
        self.model_path = model_path
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.names
        self.start_time = 0
        self.end_time = 0
    
    
    def load_model(self):

        model = YOLO(self.model_path)
        model = model.to(device='cpu')
        model.fuse()
        return model


    def test_model(self):
        # self.model.val()


    def _box_plotting(self, results):

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



class YoloTracker(YoloModel, Display):

    def __init__(self, model_path):
        super().__init__(model_path)


    def image_tracking(self, image_path: str) -> None: 
        
        # Список с результатами
        results = self.model.track(image_path, persist=True)
        # Отображение на экран
        annotated_frame = self._box_plotting(results)
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
 
        
    def stream_tracking(self, video_path: str) -> None: 

        # Открытие видео файла 
        cap = cv2.VideoCapture(video_path)        
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
                # Отображение на экране box-ов
                annotated_frame = self._box_plotting(results) 

                # Вывод fps, времени работы, эмблемы на дисплей
                self.fps_display(annotated_frame)
                self.starting_time_display(annotated_frame)
                self.logotype_display(annotated_frame)
                self.target_aiming_display(annotated_frame, 640, 360)  # x_center, y_center 

                # Вывод кадра на экран
                cv2.imshow("YOLOv8 Tracking", annotated_frame)
            
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

    print(f'Загружена модель: {tracker.model_path}')
    print(f'Используемый девайс: {tracker.device}')

    # Запуск на трекиг видеопотока
    tracker.stream_tracking(r"C:\Users\1NR_Operator_33\Downloads\танк Т90м Прорыв лучший танк в мире в бою.mp4")
