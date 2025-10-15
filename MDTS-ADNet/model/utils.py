import numpy as np
from collections import OrderedDict
import os
import sys
import glob
import cv2
import re
import torch.utils.data as data
import logging

rng = np.random.RandomState(2020)


def np_load_frame(filename, resize_height, resize_width):
    image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized


class DataLoader(data.Dataset):
    def __init__(self, video_folder_rgb, video_folder_of, transform, resize_height, resize_width, time_step=4,
                 num_pred=1):
        self.dir_rgb = video_folder_rgb
        self.dir_of = video_folder_of
        self.transform = transform
        self.videos_rgb = OrderedDict()
        self.videos_of = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step
        self._num_pred = num_pred
        self.setup()
        self.samples = self.get_all_samples()

    def setup(self):
        # Setup for RGB frames
        videos_rgb = glob.glob(os.path.join(self.dir_rgb, '*'))

        for video in sorted(videos_rgb):
            video_name = os.path.basename(video)

            self.videos_rgb[video_name] = {}
            self.videos_rgb[video_name]['path'] = video
            self.videos_rgb[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
            self.videos_rgb[video_name]['frame'].sort()  # Ensure sorting

            self.videos_rgb[video_name]['length'] = len(self.videos_rgb[video_name]['frame'])

        # Setup for optical flow frames
        videos_of = glob.glob(os.path.join(self.dir_of, '*'))

        for video in sorted(videos_of):
            video_name = os.path.basename(video)

            self.videos_of[video_name] = {}
            self.videos_of[video_name]['path'] = video
            self.videos_of[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
            self.videos_of[video_name]['frame'].sort()  # Ensure sorting

            self.videos_of[video_name]['length'] = len(self.videos_of[video_name]['frame'])

    def get_all_samples(self):
        frames_rgb = []
        frames_of = []

        for video_name in sorted(self.videos_rgb.keys()):
            for i in range(len(self.videos_rgb[video_name]['frame']) - self._time_step):
                frames_rgb.append(self.videos_rgb[video_name]['frame'][i])

        for video_name in sorted(self.videos_of.keys()):
            for i in range(len(self.videos_of[video_name]['frame']) - self._time_step):
                frames_of.append(self.videos_of[video_name]['frame'][i])

        assert len(frames_rgb) == len(frames_of), "Number of RGB frames and optical flow frames must be the same."

        return list(zip(frames_rgb, frames_of))

    def __getitem__(self, index):
        video_rgb_name = os.path.basename(os.path.dirname(self.samples[index][0]))
        video_of_name = os.path.basename(os.path.dirname(self.samples[index][1]))

        frame_name_rgb = os.path.splitext(os.path.basename(self.samples[index][0]))[0]
        frame_name_of = os.path.splitext(os.path.basename(self.samples[index][1]))[0]

        try:
            frame_name_rgb = int(re.search(r'\d+', frame_name_rgb).group())
            frame_name_of = int(re.search(r'\d+', frame_name_of).group())
        except ValueError as e:
            print(f"Error converting frame name to integer: {frame_name_rgb}, {frame_name_of}")
            raise e

        batch_rgb = []
        batch_of = []

        for i in range(self._time_step + self._num_pred):
            image_rgb = np_load_frame(self.videos_rgb[video_rgb_name]['frame'][frame_name_rgb + i],
                                      self._resize_height, self._resize_width)
            image_of = np_load_frame(self.videos_of[video_of_name]['frame'][frame_name_of + i],
                                     self._resize_height, self._resize_width)

            if self.transform is not None:
                batch_rgb.append(self.transform(image_rgb))
                batch_of.append(self.transform(image_of))

        return np.concatenate(batch_rgb, axis=0), np.concatenate(batch_of, axis=0)

    def __len__(self):
        return len(self.samples)
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()
