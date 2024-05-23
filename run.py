import warnings
import logging

from drone_tracker import YoloTracker, YoloConfig

warnings.filterwarnings('ignore')
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


if __name__ == '__main__':
    config = YoloConfig()
    tracker = YoloTracker()

    logging.info(f'Загружена модель: {tracker.model_path}')
    logging.info(f'Используемый девайс: {tracker.device}')
    
    tracker()  # Запуск трекера