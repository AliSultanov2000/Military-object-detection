import ultralytics
import torch
from collections import defaultdict
import numpy as np
import torch.backends.cudnn as cudnn
import yaml
from PIL import Image
from ultralytics import YOLO
import cv2



def load_config_data(data_path: str) -> dict: 
    """Функция для загрузки всех данных из конфигурационного файла yaml"""
    try:
        with open(data_path, 'r', encoding='utf8') as file:
            config_data = yaml.safe_load(file)
            return config_data
        
    except FileNotFoundError:
        print('Ошибка при загрузке конфигурационного файла')


# Загрузка всех переменных из конфигурационного файла
CONFIG_DATA = load_config_data('config.yaml')
print(CONFIG_DATA)



try:
    model = YOLO(CONFIG_DATA['yolov8_model'])
    model.to(device=torch.device("cuda"))
    print('Модель YOLO успешно загружена')
    print(f'Девайс для YOLO: {model.device}')
    
except Exception:
    print('Возникла ошибка при загрузке модели')



# Open the video file
video_path = r"C:\Users\1NR_Operator_33\Downloads\Поток машин на ТТК (Москва), весна-лето 2018.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(105, 0, 198), thickness=4)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
