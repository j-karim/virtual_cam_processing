from FrameProcessors import *
from MaskProcessors import *
from SequentialProcessors import *

SelfmadePencilSketchProcessor_V1 = SequentialFrameProcessor([
        # KernelFrameProcessor(),  # Laplace-kernel
        # HistogramEqualizationFrameProcessor(),
        # MTCNNFaceDetectorMaskProcessor(apply_immediate=True),
        GaussianBlurFrameProcessor(size=11),
        BGRToGrayProcessor(),
        SobelDerivativeFrameProcessor(),
        InversionProcessor()
        # BinarizeFrameProcessor(size=15, substract_mean=2.5),  # Adaptive binarization
        # DelaunayProcessor()
        ])


SelfmadePencilSketchProcessor_V2 = SequentialFrameProcessor([
        GaussianBlurFrameProcessor(size=51),
        KernelFrameProcessor(),  # Laplace-kernel
        BGRToGrayProcessor(),
        InversionProcessor(),
        GaussianBlurFrameProcessor(size=3),
        BinarizeAdaptiveFrameProcessor(size=15, substract_mean=0.1),  # Adaptive binarization
        ])


FirstParallelProcessor = SequentialFrameProcessor([
        BGRToGrayProcessor(),
        DenoiseProcessor(templateWindowSize=7, search_window_size=7),
        GaussianBlurFrameProcessor(size=5),
        ParallelFrameProcessor([[
            InversionProcessor(),
            CannyProcessor(230, 254),
            # DilationProcessor(iterations=1),
        ], [
            InversionProcessor(),
            CannyProcessor(40, 60),
        ]], AddProcessor()),

        InversionProcessor()
        ])


class PencilSketchProcessor(BaseFrameProcessor):

    def __init__(self):
        super().__init__()
        self.preprocessor = SequentialFrameProcessor([
            InversionProcessor(),
            GaussianBlurFrameProcessor(size=51)
        ])
        self.rgb2gray = BGRToGrayProcessor()

    def process_frame(self, frame):
        frame = self.rgb2gray.process_frame(frame)
        preprocessed_frame = self.preprocessor.process_frame(frame)
        return self.dodge(preprocessed_frame, frame)

    def dodge(self, front, back):
        result = front.astype('float') * 255 / (255 - back.astype('float'))
        result[result > 255] = 255
        result[back == 255] = 255
        return result.astype('uint8')



def get_inside_outside_processor():
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
    face_detection_processor = CascadeFaceDetectorMaskProcessor(negative=True, padding=(-0.1, -0.1, -0.1, -0.1))

    # define which frame processor to use
    frame_processor = SequentialFrameProcessor([
        face_detection_processor,
        # BackgroundMask(apply_immediate=True),
        BGRToGrayProcessor(),
        DenoiseProcessor(templateWindowSize=7, search_window_size=7),
        GaussianBlurFrameProcessor(size=5),

        ParallelFrameProcessor([
            inside_face_edge_processors,
            outside_face_edge_processors], AddWithMasksProcessor([None, face_detection_processor])),
        InversionProcessor()
    ])

    return frame_processor


def get_inside_outside_colored():
    processor = get_inside_outside_processor()
    processor.processor_list.append(FloodFillFaceAndBackgroundProcessor(processor.processor_list[0]))
    return processor
