from copy import deepcopy, copy

import numpy as np
import cv2
import torch

from AbstractProcessors import *

from random import sample

from SequentialProcessors import SequentialFrameProcessor
from u2net.U2NetHelper import ModelIdentifier, inference
from u2net.model import U2NET, U2NETP
from u2net.data_loader import RescaleT
from u2net.data_loader import ToTensorLab

from torchvision import transforms

from typing import Sequence





class KernelFrameProcessor(BaseFrameProcessor):
    def __init__(self, kernel=np.asarray([[0, 1, 0], [1, -4, 1], [0, 1, 0]])):  # Laplace kernel is default
        super().__init__()
        self._wait_interval = 2
        self.kernel = kernel

    def process_frame(self, frame):
        frame = cv2.filter2D(frame, -1, self.kernel)

        return frame


class GaussianBlurFrameProcessor(BaseFrameProcessor):
    def __init__(self, size=5):
        super().__init__()
        self._wait_interval = 5
        self.size = size

    def process_frame(self, frame):
        frame = cv2.GaussianBlur(frame, (self.size, self.size), 0)
        return frame


class BinarizeAdaptiveFrameProcessor(BaseFrameProcessor):
    def __init__(self, size=5, substract_mean=2):
        super().__init__()
        self._wait_interval = 2
        self.size = size
        self.substract_mean = substract_mean
        self.rgb2gray = BGRToGrayProcessor()


    def process_frame(self, frame):
        frame = self.rgb2gray.process_frame(frame)

        frame = cv2.adaptiveThreshold(frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, self.size, self.substract_mean)
        return frame





class SobelDerivativeFrameProcessor(BaseFrameProcessor):
    def __init__(self, size=3):
        super().__init__()
        self.size = size

    def process_frame(self, frame):
        sobel_x = np.absolute(cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=self.size))
        sobel_y = np.absolute(cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=self.size))

        return np.uint8(np.round((sobel_x + sobel_y)/2.0))


class HistogramEqualizationFrameProcessor(BaseFrameProcessor):
    def __init__(self):
        super().__init__()
        self._wait_interval = 5
        self.rgb2gray = BGRToGrayProcessor()

    def process_frame(self, frame):
        frame = self.rgb2gray.process_frame(frame)
        return cv2.equalizeHist(frame)



class U2NetFrameProcessor(BaseFrameProcessor):
    def __init__(self, model_identifier=ModelIdentifier.FullU2Net):

        super().__init__()
        if model_identifier == ModelIdentifier.FullU2Net:
            self._wait_interval = 5000
            model_dir = './u2net/saved_models/u2net/u2net_portrait.pth'
            print("...load U2NET---173.6 MB")
            net = U2NET()
        elif ModelIdentifier.ReducedU2et:
            self._wait_interval = 40
            model_dir = './u2net/saved_models/u2netp/u2netp.pth'
            net = U2NETP()
        else:
            raise ValueError('Model identifier unknown')



        net.load_state_dict(torch.load(model_dir, map_location=torch.device('cpu')))
        if torch.cuda.is_available():
            net.cuda()
        net.eval()

        self.model = net
        self.transform = transforms.Compose([RescaleT(512), ToTensorLab(flag=0)])



    def process_frame(self, frame):
        scale = 0.2
        frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
        pred = inference(self.model, frame)
        pred *= 255
        pred = cv2.resize(pred, (0, 0), fx=1 / scale, fy=1 / scale)

        return pred.astype('uint8')


class BGRToGrayProcessor(BaseFrameProcessor):

    def process_frame(self, frame):
        if len(frame.shape) != 3:
            return frame

        if frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame


class Gray2BGRProcessor(BaseFrameProcessor):

    def process_frame(self, frame):
        if len(frame.shape) != 2:
            return frame

        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        return frame


class InversionProcessor(BaseFrameProcessor):

    def process_frame(self, frame):
        return 255 - frame





class DelaunayProcessor(BaseFrameProcessor):
    def __init__(self, nr_points=5000, delaunay_color=(255, 255, 255)):
        super().__init__()
        self._wait_interval = 5
        self.delaunay_color = delaunay_color
        self.nr_points = nr_points


    def process_frame(self, frame):
        if not set(list(frame.flatten())) == {0, 255}:
            frame = BinarizeAdaptiveFrameProcessor().process_frame(frame)


        subdiv = cv2.Subdiv2D((0, 0, frame.shape[1], frame.shape[0]))
        points = list(np.argwhere(frame == 0))

        points = sample(points, min(self.nr_points, len(points)))

        for p in points:
            subdiv.insert((p[1], p[0]))

        frame = np.zeros_like(frame)
        self.draw_delaunay(frame, subdiv, self.delaunay_color)

        return frame

    def rect_contains(self, rect, point):
        if point[0] < rect[0]:
            return False
        elif point[1] < rect[1]:
            return False
        elif point[0] > rect[2]:
            return False
        elif point[1] > rect[3]:
            return False
        return True


    def draw_delaunay(self, img, subdiv, delaunay_color):
            triangleList = subdiv.getTriangleList()
            size = img.shape
            r = (0, 0, size[1], size[0])
            for t in triangleList:
                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])
                if self.rect_contains(r, pt1) and self.rect_contains(r, pt2) and self.rect_contains(r, pt3):
                    cv2.line(img, pt1, pt2, delaunay_color, 1)
                    cv2.line(img, pt2, pt3, delaunay_color, 1)
                    cv2.line(img, pt3, pt1, delaunay_color, 1)



class CannyProcessor(BaseFrameProcessor):
    def __init__(self, lower_thresh=None, upper_thresh=None):
        super().__init__()
        self.lower_thresh = lower_thresh
        self.upper_thresh = upper_thresh

    def process_frame(self, frame):
        if self.lower_thresh is None or self.upper_thresh is None:
            self.lower_thresh = (1 - 0.33) * np.median(frame)
            self.upper_thresh = (1 + 0.33) * np.median(frame)

        frame = cv2.Canny(frame, self.lower_thresh, self.upper_thresh)

        return frame


class OpenCloseFrameProcessor(BaseFrameProcessor):
    def __init__(self, kernel=np.ones((5, 5), np.uint8), option=cv2.MORPH_OPEN, iterations=None):
        self.kernel = kernel
        self.option = option
        self.iterations = iterations
        super().__init__()

    def process_frame(self, frame):
        closing = cv2.morphologyEx(frame, self.option, self.kernel, iterations=self.iterations)
        return closing


class DilationProcessor(BaseFrameProcessor):
    def __init__(self, kernel=np.ones((5, 5), np.uint8), iterations=None):
        self.kernel = kernel
        self.iterations = iterations
        super().__init__()

    def process_frame(self, frame):
        frame = cv2.dilate(frame, kernel=self.kernel, iterations=self.iterations)
        return frame


class ErosionProcessor(BaseFrameProcessor):
    def __init__(self, kernel=np.ones((5, 5), np.uint8), iterations=None):
        self.kernel = kernel
        self.iterations = iterations
        super().__init__()

    def process_frame(self, frame):
        frame = cv2.erode(frame, kernel=self.kernel, iterations=self.iterations)
        return frame


class DenoiseProcessor(BaseFrameProcessor):
    def __init__(self, hColor=None, templateWindowSize=None, search_window_size=None):
        super().__init__()
        self.hColor = hColor
        self.templateWindowSize = templateWindowSize
        self.search_window_size = search_window_size

    def process_frame(self, frame):
        if len(frame.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(frame, hColor=self.hColor, templateWindowSize=self.templateWindowSize,
                                                   searchWindowSize=self.search_window_size)
        elif len(frame.shape) == 2:
            return cv2.fastNlMeansDenoising(frame, templateWindowSize=self.templateWindowSize,
                                            searchWindowSize=self.search_window_size)
        else:
            raise ValueError('Image must have one or three channels')


class DrawContourProcessor(BaseFrameProcessor):
    def __init__(self, nr_colors=20):
        super().__init__()
        self.colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for i in range(nr_colors)]

    def process_frame(self, frame):
        dilated_frame = InversionProcessor().process_frame(frame)
        contours, hierarchy = cv2.findContours(dilated_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        frame = Gray2BGRProcessor().process_frame(frame)
        nr_colors = float(len(self.colors))
        new_frame = frame
        for i, color in enumerate(self.colors):
            contours_tmp = contours[int(np.floor(i / nr_colors * len(contours))):int(np.floor((i + 1.0) / nr_colors * len(contours)))]
            cv2.drawContours(new_frame, contours_tmp, -1, color)

        return new_frame


class FloodFillFaceAndBackgroundProcessor(BaseFrameProcessor):
    def __init__(self, face_detection_processor: BaseMaskProcessor,
                 color=(0, 255, 255),
                 dilation_iterations=15,
                 dilation_kernel=np.ones((3, 3), np.uint8),
                 ):
        super().__init__()
        self.face_detection_processor = face_detection_processor
        self.color = color
        self.dilation_iterations = dilation_iterations
        self.dilation_kernel = dilation_kernel

    def process_frame(self, frame):
        mask_pre_processor_background = SequentialFrameProcessor([InversionProcessor(),
                                                                  OpenCloseFrameProcessor(kernel=self.dilation_kernel,
                                                                                          option=cv2.MORPH_CLOSE,
                                                                                          iterations=self.dilation_iterations),
                                                                  OpenCloseFrameProcessor(kernel=self.dilation_kernel,
                                                                                          option=cv2.MORPH_CLOSE,
                                                                                          iterations=self.dilation_iterations),
                                                                  OpenCloseFrameProcessor(kernel=self.dilation_kernel,
                                                                                          option=cv2.MORPH_CLOSE,
                                                                                          iterations=self.dilation_iterations)
                                                                  ]
                                                                 )

        colored_frame = Gray2BGRProcessor().process_frame(frame)

        face_points = np.argwhere(self.face_detection_processor.mask > 0)
        if len(face_points) == 0:
            return colored_frame

        min_face_coordinate = np.min(face_points.flatten())
        if min_face_coordinate > 10:
            background_seed_point = np.random.randint(10, min_face_coordinate, size=2)
            background_seed_point = (background_seed_point[0], background_seed_point[1])
        else:
            background_seed_point = (50, 50)

        mask = mask_pre_processor_background.process_frame(frame)
        mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

        cv2.floodFill(colored_frame, mask=mask, seedPoint=background_seed_point, newVal=self.color, loDiff=(1, 1, 1), upDiff=(1, 1, 1), flags=cv2.FLOODFILL_FIXED_RANGE)

        return colored_frame


class AddProcessor(BaseFrameProcessor):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights

    def process_frame(self, frame_or_list):
        assert isinstance(frame_or_list, Sequence), 'Type is wrong, should be list of frames'

        if self.weights is None:
            self.weights = 1/len(frame_or_list) * np.ones(len(frame_or_list))

        assert len(frame_or_list) == len(self.weights)

        frame_list = [x.astype('float') for x in frame_or_list]
        frame = np.average(frame_list, axis=0, weights=self.weights)

        return frame.astype('uint8')


class AddWithComplementaryMasksProcessor(BaseFrameProcessor):
    def __init__(self, mask_processor, weights=None):
        super().__init__()
        self.mask_processor: BaseMaskProcessor = mask_processor
        self.weights = weights

    def process_frame(self, frame_or_list):
        assert isinstance(frame_or_list, Sequence), 'Type is wrong, should be list of frames'

        if self.weights is None:
            self.weights = 1/len(frame_or_list) * np.ones(len(frame_or_list))

        assert len(frame_or_list) == 2
        assert len(frame_or_list) == len(self.weights)
        pos_mask_processor = self.mask_processor
        neg_mask_processor = copy(self.mask_processor)
        neg_mask_processor.negative = not neg_mask_processor.negative

        frame_or_list = [pos_mask_processor.apply_mask(frame_or_list[0]), neg_mask_processor.apply_mask(frame_or_list[1])]
        frame_list = [x.astype('float') for x in frame_or_list]

        frame = np.average(frame_list, axis=0, weights=self.weights)

        return frame.astype('uint8')



class AddWithMasksProcessor(BaseFrameProcessor):
    def __init__(self, mask_processor_list, weights=None):
        super().__init__()
        self.weights = weights
        self.mask_processor_list: Sequence[BaseMaskProcessor] = mask_processor_list

    def process_frame(self, frame_or_list):
        assert isinstance(frame_or_list, Sequence), 'Type is wrong, should be list of frames'

        if self.weights is None:
            self.weights = 1 / len(frame_or_list) * np.ones(len(frame_or_list))

        assert len(frame_or_list) == len(self.mask_processor_list)
        assert len(self.weights) == len(frame_or_list)

        frame_list = [self._apply_mask(p, frame) for p, frame in zip(self.mask_processor_list, frame_or_list)]
        frame_list = [x.astype('float') for x in frame_list]
        frame = np.average(frame_list, axis=0, weights=self.weights)

        return frame.astype(np.uint8)

    @staticmethod
    def _apply_mask(p: BaseMaskProcessor, frame):
        if p is not None:
            return p.apply_mask(frame)
        else:
            return frame













