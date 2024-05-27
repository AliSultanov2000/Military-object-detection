from flask import Flask,render_template,Response
from ultralytics import YOLO
import argparse

import cv2
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = YOLO("./weights/yolov9c.pt").to(device)


app = Flask(__name__)



def generate_frames():
    cap = cv2.VideoCapture(r"./tests/car_flow.mp4")
    # Цикл по видео фрейму
    while cap.isOpened():
        # Считывание фрейма из видео
        success, frame = cap.read()

        if not success:
            break

        else:
            # Запуск трекера
            results = model.track(frame, persist=True)

            # Visualize the results on the frame
            frame = results[0].plot()
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()


            yield(b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')



@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)