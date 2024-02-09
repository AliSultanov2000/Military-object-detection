class Network(nn.Module): 
   def __init__(self, input_size, output_size): 
       super(Network, self).__init__() 

       self.seq = nn.Sequential(nn.Linear(input_size, 24),
                                nn.ReLU(),
                                nn.Linear(24, 26),
                                nn.ReLU(),
                                nn.Linear(26, output_size))
        
# Отображение результата обучения
x_epochs = np.arange(1, num_epochs + 1)


def model_train(num_epochs: int, train_loader: DataLoader, validate_loader: DataLoader):
    """Функция для обучения нейронной сети с валидацией. Метрика: Accuracy"""

    train_losses, val_losses = [], []  # Ошибка на трейне, валидации

    for epoch in range(num_epochs):
        # Тренировка за эпоху
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc='Training loop'):
            labels = labels.to(torch.long)
            # Обнуляем градиент
            optimizer.zero_grad()
    
            outputs = model(inputs)              # Предсказание
            loss = loss_fn(outputs, labels)      # Ошибка
            loss.backward()                      # Градиенты
            optimizer.step()                     # Шаг оптимизации
        
            running_loss += loss.item()       # Увеличиваем общую ошибку за эпоху
        
        # Считаем ошибку за эпоху
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Валидацию делаем только за одну эпоху
        running_loss = 0.0
        model.eval()  
        for images, labels in tqdm(validate_loader, desc='Validation loop'):
            labels = labels.to(torch.long)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

        # Считаем ошибку за эпоху
        val_loss = running_loss / len(validate_loader.dataset)
        val_losses.append(val_loss)

        # Печатаем данные на каждой эпохе, вовзращаем результаты обучения
        print(f"Epoch {epoch + 1} / {num_epochs} - Train loss: {train_loss}, Validation loss: {val_loss}")
    return train_losses, val_losses

fig, ax = plt.subplots()
ax.set(title='Процесс обучения нейросети', xlabel='Epoch', ylabel='Loss')
ax.plot(x_epochs, train_losses, label='Train')
ax.plot(x_epochs, val_losses, label='Validation')
ax.legend();

class DistanceCalculation:
    """A class to calculate distance between two objects in real-time video stream based on their tracks."""

    def __init__(self):
        """Initializes the distance calculation class with default values for Visual, Image, track and distance
        parameters.
        """

        # Visual & im0 information
        self.im0 = None
        self.annotator = None
        self.view_img = False
        self.line_color = (255, 255, 0)
        self.centroid_color = (255, 0, 255)

        # Predict/track information
        self.clss = None
        self.names = None
        self.boxes = None
        self.line_thickness = 2
        self.trk_ids = None

        # Distance calculation information
        self.centroids = []
        self.pixel_per_meter = 10

        # Mouse event
        self.left_mouse_count = 0
        self.selected_boxes = {}

        # Check if environment support imshow
        self.env_check = check_imshow(warn=True)

    def set_args(
        self,
        names,
        pixels_per_meter=10,
        view_img=False,
        line_thickness=2,
        line_color=(255, 255, 0),
        centroid_color=(255, 0, 255),
    ):
        """
        Configures the distance calculation and display parameters.

        Args:
            names (dict): object detection classes names
            pixels_per_meter (int): Number of pixels in meter
            view_img (bool): Flag indicating frame display
            line_thickness (int): Line thickness for bounding boxes.
            line_color (RGB): color of centroids line
            centroid_color (RGB): colors of bbox centroids
        """
        self.names = names
        self.pixel_per_meter = pixels_per_meter
        self.view_img = view_img
        self.line_thickness = line_thickness
        self.line_color = line_color
        self.centroid_color = centroid_color

    def mouse_event_for_distance(self, event, x, y, flags, param):
        """
        This function is designed to move region with mouse events in a real-time video stream.

        Args:
            event (int): The type of mouse event (e.g., cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, etc.).
            x (int): The x-coordinate of the mouse pointer.
            y (int): The y-coordinate of the mouse pointer.
            flags (int): Any flags associated with the event (e.g., cv2.EVENT_FLAG_CTRLKEY,
                cv2.EVENT_FLAG_SHIFTKEY, etc.).
            param (dict): Additional parameters you may want to pass to the function.
        """
        global selected_boxes
        global left_mouse_count
        if event == cv2.EVENT_LBUTTONDOWN:
            self.left_mouse_count += 1
            if self.left_mouse_count <= 2:
                for box, track_id in zip(self.boxes, self.trk_ids):
                    if box[0] < x < box[2] and box[1] < y < box[3] and track_id not in self.selected_boxes:
                        self.selected_boxes[track_id] = []
                        self.selected_boxes[track_id] = box

        if event == cv2.EVENT_RBUTTONDOWN:
            self.selected_boxes = {}
            self.left_mouse_count = 0

    def extract_tracks(self, tracks):
        """
        Extracts results from the provided data.

        Args:
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.boxes = tracks[0].boxes.xyxy.cpu()
        self.clss = tracks[0].boxes.cls.cpu().tolist()
        self.trk_ids = tracks[0].boxes.id.int().cpu().tolist()

    def calculate_centroid(self, box):
        """
        Calculate the centroid of bounding box
        Args:
            box (list): Bounding box data
        """
        return int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)

    def calculate_distance(self, centroid1, centroid2):
        """
        Calculate distance between two centroids
        Args:
            centroid1 (point): First bounding box data
            centroid2 (point): Second bounding box data
        """
        pixel_distance = math.sqrt((centroid1[0] - centroid2[0]) ** 2 + (centroid1[1] - centroid2[1]) ** 2)
        return pixel_distance / self.pixel_per_meter

    def plot_distance_and_line(self, distance):
        """
        Plot the distance and line on frame
        Args:
            distance (float): Distance between two centroids
        """
        cv2.rectangle(self.im0, (15, 25), (280, 70), (255, 255, 255), -1)
        cv2.putText(
            self.im0, f"Distance : {distance:.2f}m", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA
        )
        cv2.line(self.im0, self.centroids[0], self.centroids[1], self.line_color, 3)
        cv2.circle(self.im0, self.centroids[0], 6, self.centroid_color, -1)
        cv2.circle(self.im0, self.centroids[1], 6, self.centroid_color, -1)

    def start_process(self, im0, tracks):
        """
        Calculate distance between two bounding boxes based on tracking data
        Args:
            im0 (nd array): Image
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0
        if tracks[0].boxes.id is None:
            if self.view_img:
                self.display_frames()
            return
        self.extract_tracks(tracks)

        self.annotator = Annotator(self.im0, line_width=2)

        for box, cls, track_id in zip(self.boxes, self.clss, self.trk_ids):
            self.annotator.box_label(box, color=colors(int(cls), True), label=self.names[int(cls)])

            if len(self.selected_boxes) == 2:
                for trk_id, _ in self.selected_boxes.items():
                    if trk_id == track_id:
                        self.selected_boxes[track_id] = box

        if len(self.selected_boxes) == 2:
            for trk_id, box in self.selected_boxes.items():
                centroid = self.calculate_centroid(self.selected_boxes[trk_id])
                self.centroids.append(centroid)

            distance = self.calculate_distance(self.centroids[0], self.centroids[1])
            self.plot_distance_and_line(distance)

        self.centroids = []

        if self.view_img and self.env_check:
            self.display_frames()

        return im0

    def display_frames(self):
        """Display frame."""
        cv2.namedWindow("Ultralytics Distance Estimation")
        cv2.setMouseCallback("Ultralytics Distance Estimation", self.mouse_event_for_distance)
        cv2.imshow("Ultralytics Distance Estimation", self.im0)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            return


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

torch.save(net, 'torch_xaxa.pkl')  # Сохранили саму модель вместе с весами

dataloader = DataLoader(dataset, batch_size=4, shuffle=True, )

for idx, batch in enumerate(dataloader):
    print(idx ,batch) 

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


# ТЕСТИРОВАНИЕ ФУНКЦИИ

def model_train(num_epochs: int, train_loader: DataLoader, validate_loader: DataLoader):
    """Функция для обучения нейронной сети с валидацией. Метрика: Accuracy"""
    
    for epoch in range(num_epochs):
        running_loss, running_items, running_right = 0.0, 0.0, 0.0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch[0], train_batch[1].to(torch.long)
    
            # Обнуляем градиент
            optimizer.zero_grad()
    
            outputs = model(inputs)              # Предсказание
            loss = loss_fn(outputs, labels)      # Ошибка
            loss.backward()                      # Градиенты
            optimizer.step()                     # Шаг оптимизации
    
            # выводим статистику о процессе обучения
            running_loss += loss.item()
            running_items += len(labels)
            running_right += (labels == torch.max(outputs, 1)[1]).sum()
    
            # выводим статистику о процессе обучения
            if idx % 10 == 0:    # печатаем каждые 10 mini-batches
                model.eval()
                    
                valid_running_loss, valid_running_items, valid_running_right  = 0.0, 0.0, 0.0
                for idx, valid_batch in enumerate(validate_loader):
                    valid_inputs, valid_labels = valid_batch[0], valid_batch[1].to(torch.long)

                    valid_outputs = model(valid_inputs)              # Предсказание
                    valid_loss = loss_fn(valid_outputs, valid_labels)  # Ошибка на одном батче
                    valid_running_loss += valid_loss.item()            # Общая ошибка на всей эпохе

                    valid_running_items += len(valid_batch[1])  
                    valid_running_right += (valid_batch[1] == torch.max(valid_outputs, 1)[1]).sum()

                print(f'Epoch [{epoch + 1} / {num_epochs}]. ' \
                      f'Step [{idx + 1}/{len(train_loader)}]. ' \
                      f'Train Loss {running_loss / running_items:.3f}. ' \
                      f'Train Acc {running_right / running_items:.3f}. ' \
                      f'Valid Loss {valid_running_loss / valid_running_items:.3f} ' \
                      f'Valid Acc {valid_running_right / valid_running_items:.3f}')
                
                running_loss, running_items, running_right = 0.0, 0.0, 0.0
                    
    print('Training is finished!')
if __name__ == "__main__": 
    num_epochs = 50
    model_train(num_epochs, train_loader, validate_loader)


