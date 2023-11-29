import ultralytics
import torch
from collections import defaultdict
import numpy as np
import torch.backends.cudnn as cudnn
import yaml
from PIL import Image
from ultralytics import YOLO
import cv2





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

def objective(trial: optuna.Trial):
    # list of hyperparameters for optimization (CatBoostClassifier)
    param_distribution = {
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
        "depth": trial.suggest_int("depth", 1, 12),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
                         }

    # Smart idea! 
    if param_distribution["bootstrap_type"] == "Bayesian":
        param_distribution["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param_distribution["bootstrap_type"] == "Bernoulli":
        param_distribution["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    # The model (Here we can use Pipeline, custom classes, def function)
    model = CatBoostClassifier(**param_distribution, loss_function='MultiClass', verbose=False)

    # Cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=50)  # Set fold strategy
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='f1_macro')  # Metric results for all folds
    score = np.mean(cv_scores)  # Mean for all folds 
    std = np.std(cv_scores)  # Std by all folds

    # User attribute
    trial.set_user_attr("score", score)
    trial.set_user_attr("std", std)

    # Return the metric 
    return score


# Sampler: TPE
sampler = optuna.samplers.TPESampler(seed=42)
# DB Storage
storage_url = "sqlite:///example.db"
# study_name
study_name = "catt_optimization"
# Pruner using 
pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=2, min_early_stopping_rate=0)
#Create a study 
study = optuna.create_study(sampler=sampler,
                            direction='maximize',
                            storage=storage_url,
                            load_if_exists=True,
                            study_name=study_name,
                            pruner=pruner)


# START! 
study.optimize(objective,   # What we have to optimize
               n_trials=20,  # The number of trials 
               timeout=5000)  # The time of optimizing 

# Garbage collector 
gc.collect()
