import sys
import warnings

from detectron import ModelDetectron

warnings.filterwarnings('ignore')

sys.path.append(r"./")
from utils.logs import logger



if __name__ == '__main__':
    model = ModelDetectron()

    logger.info(f'Загружена модель: {model.model_path}')
    logger.info(f'Используемый девайс: {model.device}')

    input_video_path = r"./input_videos/test_video1.mp4"
    
    model.stream_tracking(input_video_path)  # Запуск детектора на работу
    logger.info('Завершение работы детектора')