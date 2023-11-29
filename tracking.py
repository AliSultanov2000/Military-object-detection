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

results = model.train(
   data=CONFIG_DATA['data'],  # Path to the YAML file
   imgsz=CONFIG_DATA['imgsz'],  # Image resize
   epochs=CONFIG_DATA['epochs'],  # The number of epochs
   batch=CONFIG_DATA['batch'],  # The number of batch during train
   name=CONFIG_DATA['name']  # The name of trained model
   )

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

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()


# In this case we can use Pipeline (inner and outer), ColumnTransformer, custom classes, custom def function in Pipeline, bu we selected simple example to show OptunaSearchCV

# KFold strategy
kf = KFold(n_splits=5, shuffle=True, random_state=50)


# Define the model
clf = CatBoostClassifier(verbose=False)


# Define param distribution
# IMPORTANT: IN param_distrs we can use only optuna.distribution! For instance, we can't use list, np.array and e.t.c.
param_distrs = {
                'min_data_in_leaf': optuna.distributions.IntDistribution(1, 10),
                'iterations': optuna.distributions.IntDistribution(800, 1200, 100),
                }


# OptunaSearchCV 
opt_search = optuna.integration.OptunaSearchCV(clf,
                                               param_distrs,
                                               cv=kf,
                                               scoring='f1',
                                               n_trials=10,  # Important parameters! In total we have 10 combination of hyperparameters and it's all
                                               timeout=100)  # Important parameters! In total trial time = 100 second


# Let's get started 
opt_search.fit(X_train, y_train)


# Let's look at best estimator
best_catboost_classifier = opt_search.best_estimator_

# Refit
best_catboost_classifier.fit(X_train, y_train)

# Check in test sample
best_catboost_classifier.score(X_test, y_test)
