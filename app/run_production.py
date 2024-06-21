import argparse
import sys

from flask import Flask, render_template, Response

sys.path.append('./')
from model.detectron import ModelDetectron


def parse_opt():
    """Парсер аргументов для запуска локального сервера"""

    parser = argparse.ArgumentParser()
    # Загрузка модели
    parser.add_argument('--model_type', type=str, default=r"YOLO")
    parser.add_argument('--model_weights', type=str, default=r"")
    # Входной видеопоток
    parser.add_argument('--input_video_path', type=str, default=r"")
    # Конфигурация сервера
    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--host', type=str, default=r"192.168.1.102")
    parser.add_argument('--port', type=int, default=5000)

    opt = parser.parse_args()
    return opt


def main(opt):
    """Запуск сервера с развёрнутой нейронной сетью с загруженными весами"""

    model = ModelDetectron(opt.model_type, opt.model_weights)

    app = Flask(__name__)

    @app.route('/')
    def index():
        return render_template('index.html')
    

    @app.route('/video')
    def video():
        return Response(model(opt.input_video_path),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    
    
    # Запуск сервера
    app.run(debug=opt.debug,
            port=opt.port)


if __name__ == '__main__':
    opt = parse_opt()  
    main(opt)
