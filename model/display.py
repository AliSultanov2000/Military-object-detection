import cv2
import sys

from dataclasses import dataclass
sys.path.append('./')
from utils.video_utils import get_bbox_center, get_bbox_width


@dataclass 
class DisplayConfig:
    #-----------------Для отображения доп.параметров-----------------
    font_face: int = cv2.FONT_HERSHEY_COMPLEX
    font_scale: float = 0.8
    thickness: int = 1
    pointer_thickness: int = 2
    linetype: int = cv2.LINE_AA
    #-----------------Для отображения bounding boxes-----------------
    box_text_color: tuple = (255, 255, 255)
                          
    margin: int = 1
    box_thickness: int = 2
    box_text_font_face: int = cv2.FONT_HERSHEY_COMPLEX
    box_text_font_scale: float = 0.5
    box_text_thickness: int = 1
    box_text_linetype: int = cv2.LINE_AA
    #-----------------Для отображения эллипса объекта----------------
    ellipse_angle: float = 0.0
    ellipse_start_angle: int = -45
    ellipse_end_angle: int = 235
    ellipse_color: tuple = (48, 45, 155)
    ellipse_thickness: int = 2
    ellipse_linetype: int = cv2.LINE_AA
    #----------------------------------------------------------------

    

class Display(DisplayConfig):
    """Класс для отображения работы трекера на экран"""

    def draw_boxing(self, im0, bbox, text, color):
        """Отображение bounding box для распознанного объекта"""
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        # Отображение Bounding box
        cv2.rectangle(im0,
                      (x1, y1), (x2, y2),
                      color,
                      self.box_thickness)
                
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
                    color,
                    -1)
        
        # Отображение id, cls, conf 
        cv2.putText(im0,  text, (x1, y1),
            self.box_text_font_face,
            self.box_text_font_scale,
            self.box_text_color,
            self.box_text_thickness,
            self.box_text_linetype)
        
    
    def draw_ellipse(self, im0, bbox):
        """Отображение эллипса для обнаруженного вооруженного человека"""
        x_center, _ = get_bbox_center(bbox)
        width = get_bbox_width(bbox)
        y2 = int(bbox[3])

        cv2.ellipse(
            im0,
            center=(int(x_center), y2),
            axes=(int(width // 2), int(0.08 * width)),
            angle = self.ellipse_angle,
            startAngle = self.ellipse_start_angle,
            endAngle= self.ellipse_end_angle,
            color = self.ellipse_color,
            thickness=self.ellipse_thickness,
            lineType=self.ellipse_linetype
        )
