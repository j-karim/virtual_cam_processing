from AbstractProcessors import *
from mtcnn.mtcnn import MTCNN
import cv2


class StupidMaskProcessor(BaseMaskProcessor):
    def __init__(self):
        super().__init__()
        self._wait_interval = 0

    def calculate_mask(self, frame):
        mask = np.zeros((frame.shape[0], frame.shape[1]))
        mask[:, :int(frame.shape[1]/2)] = 1
        return mask


class CascadeFaceDetectorMaskProcessor(BaseMaskProcessor):
    def __init__(self, apply_immediate=False, negative=False, padding=(0, 0, 0, 0)):
        super().__init__(apply_immediate, negative)
        self.padding = padding

        face_detector_path = r'u2net/saved_models/face_detection_cv2/haarcascade_frontalface_default.xml'
        self.face_detector = cv2.CascadeClassifier(face_detector_path)
        self.roi_boxes = None

    def calculate_mask(self, frame):
        self.roi_boxes = self.face_detector.detectMultiScale(frame)
        mask = np.zeros((frame.shape[0], frame.shape[1]))

        for (ex, ey, ew, eh) in self.roi_boxes:
            padding = self.padding
            ex_padded = int(ex - padding[0] * eh)
            ey_padded = int(ey - padding[1] * eh)
            ex2_padded = int(ex + (1 + padding[2]) * ew)
            ey2_padded = int(ey + (1 + padding[3]) * eh)

            cv2.rectangle(mask, (ex_padded, ey_padded), (ex2_padded, ey2_padded), 1, -1)

        return mask


class MTCNNFaceDetectorMaskProcessor(BaseMaskProcessor):

    def __init__(self, apply_immediate=False, negative=False):
        super().__init__(apply_immediate, negative)
        self.face_detector = MTCNN()

    def calculate_mask(self, frame):
        faces = self.face_detector.detect_faces(frame)
        mask = np.zeros((frame.shape[0], frame.shape[1]))

        roi_boxes = [x['box'] for x in faces]
        for (ex, ey, ew, eh) in roi_boxes:
            ey2 = int(ey + eh)
            ey = int(ey - 0.5 * eh)
            cv2.rectangle(mask, (ex, ey), (ex + ew, ey2), 1, -1)

        return mask


class BackgroundMask(BaseMaskProcessor):
    def __init__(self, apply_immediate=False, negative=False):
        super().__init__(apply_immediate, negative)
        self.foreground_detector = cv2.createBackgroundSubtractorMOG2()

    def calculate_mask(self, frame):
        mask = self.foreground_detector.apply(frame)

        mask = np.uint8(mask/255.0)

        return mask
