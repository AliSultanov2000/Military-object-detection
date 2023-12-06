import cv2
import cvzone
import numpy as np
from time import time

# Важная ссылка
#  YOUTUBE: Live tracking on real FPV drone video | Autonomous Drone Object Tracking OpenCV Python   - отображение указателя 


class Display:
    """Класс для отображения дополнительной информации о работе трекера на экран"""

    def fps_display(self, im0) -> None:
        """Отображение на экран FPS"""
        self.end_time = time()
        fps = 1 / np.round(self.end_time - self.start_time, 2)
        text = f'FPS: {int(fps)}'
        cv2.putText(im0, text, (1140, 40), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)
    

    def starting_time_display(self, im0) -> None: 
        """Отображение на экран времени работы"""
        cv2.putText(im0, 'Время', (1140, 80), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), 2)

    
    def target_aiming_display(self, im0, x_center: int, y_center: int) -> None:
        """Указатель на цель при отработке трекера"""
        cv2.circle(im0, (x_center, y_center), 5, (0, 255, 0), -1)
        gap = x_center // 32
        line_length = x_center // 10
        cv2.line(im0, (x_center - gap - line_length, y_center), (x_center - gap, y_center), (0, 255, 0), 2)
        cv2.line(im0, (x_center + gap, y_center), (x_center + gap + line_length, y_center), (0, 255, 0), 2)  
        cv2.line(im0, (x_center, y_center + gap), (x_center, y_center + gap + line_length // 2),  (0, 255, 0), 2)


    def logotype_display(self, im0) -> None:
        """Отображение логотипа"""
        img2 = cv2.imread(r"D:\flag.png")
        img2 = cv2.resize(img2, (48, 48))

        rows, cols, channels = img2.shape
        roi = im0[0: rows, 0: cols]
        # Now create a mask of logo and create its inverse mask also
        img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)

        mask_inv = cv2.bitwise_not(mask)
        # Now black-out the area of logo in ROI
        img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        # Take only region of logo from logo image.
        img2_fg = cv2.bitwise_and(img2, img2, mask = mask + 10)
        # Put logo in ROI and modify the main image
        dst = cv2.add(img1_bg, img2_fg)
        im0[0: rows, 0: cols] = dst

        cv2.imshow("YOLOv8 Tracking", im0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



img = cv2.imread(r"D:\gg.PNG")
print(img)
display = Display()
display.logotype_display(img)
