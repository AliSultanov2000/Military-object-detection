import cv2


def get_frame_size(cap):
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return frame_width, frame_height

def set_frame_size(cap, frame_width: int, frame_height: int):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)


def get_bbox_center(bbox) -> tuple:
    return (float((bbox[0] + bbox[2]) // 2), float((bbox[1]) + bbox[3]) // 2)


def get_bbox_width(bbox):
    return bbox[2] - bbox[1]
