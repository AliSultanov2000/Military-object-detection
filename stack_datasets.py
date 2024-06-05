import shutil
import time

from file_image_rename import DatasetImageRename


class DatasetStack:
    def __init__(self):
        self.split_data = ['train', 'test', 'valid']
        self.annotated_images_name = 'images'
        self.annotated_labels_name = 'labels'

        self.first_img_name = 'img'
        self.second_img_name = 'mg'


    def file_moves(self, source_file_path: str, destination_file_path: str):
        """
           Функция для переноса объектов из папки в другую папку
           source_file_path: путь до папки откуда забираем данные
           destination_file_path: путь до папки куда сохраняем файлы

        """

        for data in self.split_data:  # Пробегаемся по train, test, valid
            # Путь до папки, где хранятся data для переноса в другую папку
            imgs_source = f'{source_file_path}\{data}\{self.annotated_images_name}'
            labels_source = f'{source_file_path}\{data}\{self.annotated_labels_name}'

            # Путь до папки, где хранятся data куда будем переносить другую папку 
            imgs_destination = f'{destination_file_path}\{data}\{self.annotated_images_name}'
            labels_destination = f'{destination_file_path}\{data}\{self.annotated_labels_name}'

            shutil.copytree(imgs_source, imgs_destination, dirs_exist_ok=True)
            shutil.copytree(labels_source, labels_destination, dirs_exist_ok=True)

        print("Перенос всех файлов успешно осуществлен")


    def stack_two_datasets(self, first_dataset_path: str, second_dataset_path: str) -> None:
        """
           Объединение двух датасетов по train, test, valid
           first_dataset_path: абсолютный путь до первого датасета
           second_dataset_path: абсолютный путь до второго датасета
        """
        start_time = time.time()
        img_rename = DatasetImageRename()

        img_rename._all_change_names(first_dataset_path, self.first_img_name) 
        img_rename._all_change_names(second_dataset_path, self.second_img_name)
        
        self.file_moves(second_dataset_path, first_dataset_path)
        print(f'Время объединения двух датасетов: {time.time() - start_time} сек.')



if __name__ == '__main__':
    stack_dataset = DatasetStack()
    # stack_dataset.stack_two_datasets(r"D:\Russian License Plates Detector.v3i.yolov8", r"D:\Swimming Pools.v6i.yolov8")