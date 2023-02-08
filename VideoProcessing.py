import cv2
import os

import pyvirtualcam as pvc

from ExampleProcessors import *
from FrameProcessors import *
from MaskProcessors import *

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def display_processed_video():
    outside_face_edge_processors = [
            InversionProcessor(),
            CannyProcessor(30, 150),
            OpenCloseFrameProcessor(option=cv2.MORPH_CLOSE),
            DilationProcessor(),
        ]
    inside_face_edge_processors = [
            InversionProcessor(),
            CannyProcessor(15, 50),
        ]
    face_detection_processor = CascadeFaceDetectorMaskProcessor(negative=True, padding=(-0.2, -0.2, -0.2, -0.2))

    # define which frame processor to use
    frame_processor = SequentialFrameProcessor([
        face_detection_processor,
        BGRToGrayProcessor(),
        DenoiseProcessor(templateWindowSize=7, search_window_size=7),
        GaussianBlurFrameProcessor(size=5),

        ParallelFrameProcessor([
            inside_face_edge_processors,
            outside_face_edge_processors], AddWithMasksProcessor([None, face_detection_processor])),
        InversionProcessor(),
        FloodFillFaceAndBackgroundProcessor(face_detection_processor)
        ])


    cv2.namedWindow("preview proccessed")  # open a video preview window
    cv2.namedWindow("preview")  # open a video preview window
    vc = cv2.VideoCapture(1)  # initialize a video capture object (this object can read camera status and frames)

    if vc.isOpened():  # try to get the first frame
        ret_val, frame = vc.read()  # read the first frame
    else:
        ret_val = False

    with pvc.Camera(width=1280, height=720, fps=20, fmt=pvc.PixelFormat.BGR) as cam:
        print(f'Using virtual camera: {cam.device}')
        frame = np.zeros((cam.height, cam.width, 3), np.uint8)  # RGB
        processed_frame = frame
        while ret_val:  # ret_val is false if camera is disconnected
            cv2.imshow("preview processed", processed_frame)  # show the current frame
            cv2.imshow("preview", frame)  # show the current frame
            ret_val, frame = vc.read()  # read the next status and frame
            processed_frame = frame_processor.process_frame(frame)  # process frame
            key = cv2.waitKey(frame_processor.wait_interval)  # wait a few milliseconds
            if key == 27:  # exit on ESC
                break
            cam.send(processed_frame)
            cam.sleep_until_next_frame()

    vc.release()  # disconnect from the camera
    cv2.destroyWindow("preview")  # close the window




if __name__ == '__main__':
    display_processed_video()
