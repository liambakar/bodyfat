import cv2
import os
import pandas as pd
import torch


def play_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Video", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def validate_dataset(df: pd.DataFrame, root: str) -> None:
    video_folder = os.path.join(root, 'videos')
    for path in df['video_path']:
        vid_path = os.path.join(video_folder, path)
        if not (os.path.exists(vid_path) and os.path.isfile(vid_path)):
            raise ValueError(
                f'Dataset contains a nonexistent path: {vid_path}')


def get_dataset(csv_path: str = 'dataset/dataset.csv') -> pd.DataFrame:
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    csv_path = os.path.join(root, csv_path)

    df = pd.read_csv(csv_path)
    validate_dataset(df, root)
    return df
