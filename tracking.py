import torch
from collections import defaultdict
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2



class YoloModel:
    """Базовый класс YOLOv8"""

    def __init__(self, model_path):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model_path = model_path

        self.model = self.load_model()

        self.CLASS_NAMES_DICT = self.model.model.names

    
    def load_model(self):
        """Загрузка модели YOLOv8"""

        model = YOLO(self.model_path)

        model = model.to(device='cpu')

        # model.fuse()

        return model


    def save_model(self):
        """Сохранение модели YOLOv8"""
        pass


    def test_model(self):
        """Запуск на тестирование модели"""
    

 
class YoloTracker(YoloModel):
    """Класс YOLOv8 для задачи трекинга объектов"""

    def __init__(self, model_path):
        super().__init__(model_path)



    def image_tracking(self, image_path: str) -> None: 
        """Предсказание модели на изображении"""
        results = self.model.track(image_path)  # Cписок с результатами
        # Отображение результатов
        for result in results:
            im_array = result.plot()  # Список BGR в numpy array 
            im = Image.fromarray(im_array[..., ::-1])  # Переводим в RGB PIL image
            im.show()  # Отображение изображения


    def stream_tracking(self, video_path: str) -> None: 
        """Предсказание модели на видеопотоке"""
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        # Store the track history
        track_history = defaultdict(lambda: [])

        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()
        
            if success:
                try:
                    # Run YOLOv8 tracking on the frame, persisting tracks between frames
                    results = self.model.track(frame, persist=True)
    
                    # Get the boxes and track IDs
                    boxes = results[0].boxes.xywh.cpu()
                    track_ids = results[0].boxes.id.int().cpu().tolist()
            
            
                    # Visualize the results on the frameq
                    annotated_frame = results[0].plot()
            
                    # Plot the tracks
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box
                        x, y, w, h = int(x), int(y), int(w), int(h)
                        track = track_history[track_id]
                        track.append((float(x), float(y)))  # x, y center point
                        if len(track) > 30:  # retain 90 tracks for 90 frames
                            track.pop(0)
            
                        # Draw the tracking lines
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 255, 0), thickness=2)
            
                        # Draw the circle for bounding boxes
                        cv2.circle(annotated_frame, (x, y), 5, (0, 0, 255), -1)
            
                    # Display the annotated frame
                    cv2.imshow("YOLOv8 Tracking", annotated_frame)
                
                except AttributeError:
                    cv2.imshow("YOLOv8 Tracking", frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the end of the video is reached
                break

        cap.release()
        cv2.destroyAllWindows()
