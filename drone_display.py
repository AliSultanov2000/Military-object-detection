import cv2
import cvzone
import numpy as np
from time import time
import datetime




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
