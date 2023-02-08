import cv2
import numpy as np

from abc import ABC, abstractmethod


class BaseFrameProcessor(ABC):
    def __init__(self):
        self._wait_interval = 2

    @abstractmethod
    def process_frame(self, frame_or_list):
        pass


    @property
    def wait_interval(self):
        return max(self._wait_interval, 20)

    @wait_interval.setter
    def wait_interval(self, value):
        self._wait_interval = value


class BaseMaskProcessor(BaseFrameProcessor, ABC):
    def __init__(self, apply_immediate=False, negative=False):
        super().__init__()
        self.apply_immediate = apply_immediate
        self.negative = negative
        self.mask = None
        self.negative_mask = None


    @abstractmethod
    def calculate_mask(self, frame):
        pass

    def process_frame(self, frame):
        self.mask = np.uint8(self.calculate_mask(frame))
        self.negative_mask = np.uint8(1 - self.mask)
        if self.apply_immediate:
            frame = self.apply_mask(frame)
        return frame

    def apply_mask(self, frame):
        if self.negative:
            mask = self.negative_mask
        else:
            mask = self.mask

        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        return masked_frame

