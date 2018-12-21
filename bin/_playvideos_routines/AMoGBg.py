import cv2

class AdaptiveMoGBg(object):
    def __init__(self):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=1000, detectShadows=False)

    def __call__(self, cl_frames):
        new_cl_frames = []
        for cl, f in cl_frames:
            if f is None:
                new_cl_frames.append((cl, f))
            else:
                new_cl_frames.append((cl, self.bg.apply(f)))
        return new_cl_frames

def frame_processor():
    return AdaptiveMoGBg()
