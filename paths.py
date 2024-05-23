from pathlib import Path


def latest_video_file_num(yolo_config: dict) -> int:
    saved_video_folder = f"{yolo_config.saved_video_folder}"
    saved_videos_paths = list(Path(saved_video_folder).iterdir())
    if len(saved_videos_paths) == 0:
        return str(0)
    
    return str(len(saved_videos_paths))
    