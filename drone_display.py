import cv2
import cvzone
import numpy as np
import datetime

from time import time
from dataclasses import dataclass


@dataclass 
class DisplayConfig:
    #-----------------Для отображения доп.параметров-----------------
    logo_path: str = r"images/russian_flag.PNG"
    logo_size: tuple = (70, 60)
    logo_pos: tuple = (10, 10)

    color: tuple = (0, 255, 0)
    font_face: int = cv2.FONT_HERSHEY_COMPLEX
    font_scale: float = 0.8
    thickness: int = 1
    pointer_thickness: int = 2
    linetype: int = cv2.LINE_AA
    #-----------------Для отображения bounding boxes-----------------
    box_text_color: tuple = (255, 255, 255)

    box_colors: tuple = ((255, 56, 56), (255, 157, 151), (255, 112, 31),
                         (255, 178, 29), (207, 210, 49), (72, 249, 10),
                         (146, 204, 23), (61, 219, 134), (26, 147, 52),
                         (0, 212, 187), (44, 153, 168), (0, 194, 255),
                         (52, 69, 147), (100, 115, 255), (0, 24, 236),
                         (132, 56, 255), (82, 0, 133), (203, 56, 255),
                         (255, 149, 200), (255, 55, 199))

    margin: int = 1
    box_thickness: int = 2
    box_text_font_face: int = cv2.FONT_HERSHEY_SIMPLEX
    box_text_font_scale: float = 0.5
    box_text_thickness: int = 1
    box_text_linetype: int = cv2.LINE_AA




class Display(DisplayConfig):
    """Класс для отображения работы трекера на экран"""

    def __init__(self, frame_width: int, frame_height: int):
        self.start_time = self.end_time = 0
        self.FPS_START_TIME = time()

        self.frame_width = frame_width
        self.frame_height = frame_height
        self.x_text_info = int(frame_width // 1.35) 



    def working_time_display(self, im0) -> None: 
        """Отображение времени работы на экран"""

        self.FPS_CURRENT_TIME = time()
        working_time = str(datetime.timedelta(seconds=round(self.FPS_CURRENT_TIME - self.FPS_START_TIME)))

        cv2.putText(im0, f'Время работы: {working_time}', (self.x_text_info, 25),
                    self.font_face,
                    self.font_scale,
                    self.color,
                    self.thickness,
                    self.linetype)


    def fps_display(self, im0) -> None:
        """Отображение FPS на экране"""

        self.end_time = time()
        fps = 1 / np.round(self.end_time - self.start_time, 2)

        cv2.putText(im0, f'FPS: {int(fps)}', (self.x_text_info, 60),
                    self.font_face,
                    self.font_scale,
                    self.color,
                    self.thickness,
                    self.linetype)
    


    def target_aiming_display(self, im0) -> None:
        """Отображение прицела на экране"""

        # Отображение линии наведения дрона
        x_aim, y_aim = self.frame_width // 2, self.frame_height // 2
        gap = int(x_aim // 32)
        line_length = int(x_aim // 11)

        cv2.circle(im0, (x_aim, y_aim), 5, self.color, -1)

        cv2.line(im0,
                (x_aim - gap - line_length, y_aim),
                (x_aim - gap, y_aim),
                self.color,
                self.pointer_thickness)
        
        cv2.line(im0,
                (x_aim + gap, y_aim),
                (x_aim + gap + line_length, y_aim),
                self.color,
                self.pointer_thickness)  
        
        cv2.line(im0,
                (x_aim, y_aim + gap),
                (x_aim, y_aim + gap + line_length // 2),
                self.color,
                self.pointer_thickness)

        # Отображение прямоугольника наведения
        pointer_box_w = int(x_aim // 3.2)  
        pointer_box_h = int(y_aim / 2.4)  
        cvzone.cornerRect(im0, (x_aim - pointer_box_w // 2, y_aim - pointer_box_h // 2, pointer_box_w, pointer_box_h), rt=0)

    

    def logotype_display(self, im0) -> None:
        """Отображение логотипа на экране"""
        
        img_front = cv2.imread(self.logo_path, cv2.IMREAD_UNCHANGED)
        img_front = cv2.resize(img_front, self.logo_size)
        cvzone.overlayPNG(im0, img_front, self.logo_pos)



    def project_name_display(self, im0) -> None: 
        """Отображение названия трекера на экране"""

        cv2.putText(im0, r'Трекер ВТ', ((self.frame_width // 2) - 50, 25),
                    self.font_face,
                    self.font_scale,
                    self.color,
                    self.thickness, 
                    self.linetype)



    def priority_aim_display(self, im0, xyxys, idx_target: int, check_obj_idx: int) -> None:
        """Отображение приоритетной цели на экране"""
        x1, y1, x2, y2 = xyxys[idx_target][0], xyxys[idx_target][1], xyxys[idx_target][2], xyxys[idx_target][3]
        x_c = (x1 + x2) // 2
        y_c = (y1 + y2) // 2

        # Отображение линии наведения приоритетной цели
        cv2.line(im0, (x_c, 0), (x_c, y1 - y_c // 20),
                self.color,
                self.pointer_thickness)
        
        cv2.line(im0, (x_c, y2 + y_c // 20),
                 (x_c, self.frame_height),
                 self.color,
                 self.pointer_thickness)
        
        cv2.line(im0, (0, y_c), (x1 - x_c //20, y_c),
                 self.color,
                 self.pointer_thickness)
        
        cv2.line(im0, (x2 + x_c // 20, y_c),
                 (self.frame_width, y_c),
                 self.color,
                 self.pointer_thickness)
        
        # Отображение на экране приоритетной цели 
        cv2.putText(im0, f'id: {check_obj_idx}; x:{x_c}; y:{y_c}', (self.x_text_info, 95),
                    self.font_face,
                    self.font_scale,
                    self.color,
                    self.thickness,
                    self.linetype)

    
    def draw_boxing(self, im0, idx, xyxys, ids, cls, conf, names):
        """Отображение bounding box для распознанного объекта"""
        x1, y1, x2, y2 = xyxys[idx][0], xyxys[idx][1], xyxys[idx][2], xyxys[idx][3]
        # Отображение Bounding box
        cv2.rectangle(im0,
                      (x1, y1), (x2, y2),
                      self.box_colors[cls[idx]],
                      self.box_thickness)
        
        text = f"id:{ids[idx]} {names[cls[idx]]} {conf[idx]:.2f}"
        
        text_size = cv2.getTextSize(text,
                        self.box_text_font_face,
                        self.box_text_font_scale,
                        self.box_text_thickness)[0]
        
        rect_x1 = x1 - self.margin
        rect_y1 = y1 - text_size[1] - self.margin 
        rect_x2 = x1 + text_size[0] + self.margin
        rect_y2 = y1 + self.margin

        # Отображение прямоугольника для текста
        cv2.rectangle(im0,
                    (rect_x1, rect_y1),
                    (rect_x2, rect_y2),
                    self.box_colors[cls[idx]],
                    -1)
        
        # Отображение id, cls, conf 
        cv2.putText(im0,  text, (x1, y1),
            self.box_text_font_face,
            self.box_text_font_scale,
            self.box_text_color,
            self.box_text_thickness,
            self.box_text_linetype)