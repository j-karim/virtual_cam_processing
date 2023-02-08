from AbstractProcessors import *
from typing import Sequence


class SequentialFrameProcessor(BaseFrameProcessor):

    def __init__(self, processor_list: Sequence[BaseFrameProcessor]):
        super().__init__()
        self.processor_list = processor_list
        self._wait_interval = min(sum([x.wait_interval for x in processor_list]), 10)

    def process_frame(self, frame):
        for p in self.processor_list:
            frame = p.process_frame(frame)

        return frame


class ParallelFrameProcessor(BaseFrameProcessor):
    def __init__(self, processor_list: Sequence[Sequence[BaseFrameProcessor]], frame_combine_processor: BaseFrameProcessor):
        super().__init__()
        self.processor_list = [SequentialFrameProcessor(p) for p in processor_list]
        self.frame_combine_processor = frame_combine_processor

    def process_frame(self, frame):
        frames = []
        for seq_p in self.processor_list:
            frames.append(seq_p.process_frame(frame))

        return self.frame_combine_processor.process_frame(frames)

