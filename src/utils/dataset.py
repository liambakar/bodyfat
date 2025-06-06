from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import torchvision
import os


class VideoDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.labels = df['bf_percent']
        self.video_paths = df['video_path']

        video_root = os.path.dirname(
            os.path.dirname(os.path.dirname(__file__))
        )
        self.video_root = os.path.join(
            video_root,
            'dataset',
            'videos',
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index) -> tuple[torch.Tensor, float]:
        video_path = os.path.join(
            self.video_root, self.video_paths.iloc[index])
        # video: (batch_size, num_frames, in_channels, H_orig, W_orig)
        video_frames, _, _ = torchvision.io.read_video(
            video_path, pts_unit='sec', output_format='TCHW')
        return video_frames, self.labels.iloc[index]

    def get_dataloader(self, batch_size, shuffle=True) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
