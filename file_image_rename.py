from pathlib import Path


class DatasetImageRename:
    def __init__(self):
        self.split_data = ['train', 'test', 'valid']
        self.annotated_images_name = 'images'
        self.annotated_labels_name = 'labels'


    def _change_names(self, annoted_data_folder_path: str, img_name: str) -> None:
        """
           Изменение названия images и labels в размеченной папке
           annotated_data_folder_path: абсолютный путь до папки где хранятся выборки с images, labels
           img_name: базовое имя для названий сохранённых изображений, их меток
        """

        images_abs_paths = list(Path(f"{annoted_data_folder_path}\{self.annotated_images_name}").iterdir())
        labels_abs_paths = list(Path(f"{annoted_data_folder_path}\{self.annotated_labels_name}").iterdir())

        assert len(images_abs_paths) == len(labels_abs_paths), 'Кол-во images не равно кол-ву labels!'

        for i in range(len(list(images_abs_paths))):
            try:
                img_abs_path = Path(images_abs_paths[i])
                label_abs_path = Path(labels_abs_paths[i])
    
                new_name = f'{img_name}{i}' # Новое название как для images, так и для labels
    
                # Обрабатываем файл image
                img_parent = str(img_abs_path.parent)  # Путь до папки где лежит объект
                img_extension = str(img_abs_path.suffix)
                # Изменение названия файла image
                img_abs_path.rename(f'{img_parent}\{new_name}{img_extension}')
                # Обрабатываем файл label
                txt_parent = str(label_abs_path.parent)
                txt_extension = str(label_abs_path.suffix)
                # Изенение названия файла label
                label_abs_path.rename(f'{txt_parent}\{new_name}{txt_extension}')
           
            except FileExistsError:
                continue
            

    def _all_change_names(self, annotated_folder_path: str, img_name: str) -> None:
        """
        Изменение названия images и labels во всех размеченных папках train, test, valid.
        annotated_folder_path: путь до папки, где хранятся train, test, valid
        img_name: базовое имя для названий сохранённых изображений, их меток
        
        """

        for data in self.split_data:  # Пробегаемся по train, test, valid
            annotated_data_folder_path = f'{annotated_folder_path}\{data}'
            self._change_names(annotated_data_folder_path, img_name)
        
        print(f'Все данные из {annotated_folder_path} успешно обработаны')