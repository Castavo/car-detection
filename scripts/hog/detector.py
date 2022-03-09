from .classifier import HOGClassifier
from tqdm.auto import tqdm
import numpy as np
from .bounding_box_utils import check_intersection

class Detector:
    def __init__(self, classifier: HOGClassifier, nms_mode="surface") -> None:
        self.classifier = classifier
        self.nms_mode = nms_mode 

    def detect(self, image, stride, window_widths, window_ratios):
        detections = []

        H, W = image.shape[:2]
        progress = tqdm(total=len(window_widths)*len(window_ratios))
        for width in window_widths:
            for ratio in window_ratios:
                progress.update()
                window_size = (width, int(width * ratio))
                for i in range(0, H - window_size[1], stride):
                    for j in range(0, W - window_size[0], stride):
                        sub_image = image[i: i + window_size[1], j: j + window_size[1]]
                        if self.classifier.predict(sub_image):
                            detections.append((j, i, *window_size))
        progress.close()
        return detections

    def nms(self, detections, threshold=.75):
        res = []
        if self.nms_mode == "surface":
            sorting_criterion = [detec[2] * detec[3] for detec in detections]
        else:
            # need to store the classifier decision function in the detection
            pass
        sorting = np.argsort(sorting_criterion)
        detections = np.array(detections)[sorting]
        chosen_ones = [detections[0]]
        for detection in detections[1:]:
            if not check_intersection(chosen_ones, detection, threshold):
                chosen_ones.append(detection)
        return chosen_ones