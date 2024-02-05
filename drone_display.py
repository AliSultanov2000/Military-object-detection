import cv2
import cvzone
import numpy as np
from time import time
import datetime


class Display:
    """Класс для отображения дополнительной информации о работе трекера на экран"""

    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.START_TIME = time()


    def working_time_display(self, im0) -> None: 
        """Отображение на экран времени работы"""

        self.CURRENT_TIME = time()
        working_time = str(datetime.timedelta(seconds=round(self.CURRENT_TIME - self.START_TIME)))
        cv2.putText(im0, f'Время работы: {working_time}', (950, 25), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)


    def fps_display(self, im0) -> None:
        """Отображение на экран FPS"""
        
        self.end_time = time()
        fps = 1 / np.round(self.end_time - self.start_time, 2)
        text = f'FPS: {int(fps)}'
        cv2.putText(im0, text, (1163, 60), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
    

    def target_aiming_display(self, im0, x_center: int, y_center: int) -> None:
        """Указатель на цель при отработке трекера"""
        
        # Отображение линии наведения
        gap = int(x_center // 32)
        line_length = int(x_center // 11)
        cv2.circle(im0, (x_center, y_center), 5, (0, 255, 0), -1)
        cv2.line(im0, (x_center - gap - line_length, y_center), (x_center - gap, y_center), (0, 255, 0), 2)
        cv2.line(im0, (x_center + gap, y_center), (x_center + gap + line_length, y_center), (0, 255, 0), 2)  
        cv2.line(im0, (x_center, y_center + gap), (x_center, y_center + gap + line_length // 2),  (0, 255, 0), 2)

        # Отображение прямоугольника наведения
        pointer_box_w = int(x_center // 3.2)  
        pointer_box_h = int(y_center / 2.4)  
        cvzone.cornerRect(im0, (x_center - pointer_box_w // 2, y_center - pointer_box_h // 2, pointer_box_w, pointer_box_h), rt=0)
    

    def logotype_display(self, im0) -> None:
        """Отображение логотипа. Логотип должен быть только в PNG"""
        
        img_front = cv2.imread(r"D:\russian_flag.PNG", cv2.IMREAD_UNCHANGED)
        img_front = cv2.resize(img_front, (64, 48))
        cvzone.overlayPNG(im0, img_front, pos=[10, 10])


    def project_name_display(self, im0, x_center: int, y_center: int) -> None: 
        """Отображение названия трекера"""

        text = 'Трекер ВТ'
        cv2.putText(im0, text, (x_center - 50, 25), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)


class MyDataSet(Dataset):
    def __init__(self, data_dir, transform=None): 
        self.data = ImageFolder(data_dir, transform=transform)

    def __len__(self):
        return self.data

    def __getitem__(self, idx): 
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()   # Создаём объект базового класса nn.Module
        self.fc1 = nn.Linear(10, 15)  
        self.fc2 = nn.Linear(15, 20)
        self.fc3 = nn.Linear(20, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)
    

# Создаём экземпляр класса
net = Net()

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)  # Оптимизатор
criterion = nn.NLLLoss()  # Функция потерь отрицательного логарифмического подобия
