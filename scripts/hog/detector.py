from .classifier import HOGClassifier
from tqdm.auto import tqdm
import numpy as np
from .bounding_box_utils import check_intersection

class Detector:
    def __init__(self, classifier: HOGClassifier, nms_mode="surface") -> None:
        self.classifier = classifier
        self.nms_mode = nms_mode 

    def detect(self, image, stride, window_widths, window_ratios, verbose=True, height_range=None):
        detections = []
        decisions = []

        H, W = image.shape[:2]
        height_range = height_range or [0, H]
        if verbose: progress = tqdm(total=len(window_widths)*len(window_ratios))
        for width in window_widths:
            for ratio in window_ratios:
                if verbose: progress.update()
                window_size = (width, int(width * ratio))
                for i in range(height_range[0], height_range[1] - window_size[1], stride):
                    for j in range(0, W - window_size[0], stride):
                        sub_image = image[i: i + window_size[1], j: j + window_size[1]]
                        decision = self.classifier.predict(sub_image, return_decision=True)[0]
                        if decision > 0:
                            detections.append((j, i, *window_size))
                            decisions.append(decision)
        if verbose: progress.close()
        return detections, decisions

    def nms(self, detections, threshold=.75, thresholding_method="min", confidences=None):
        if len(detections) == 0:
            return []
        if self.nms_mode == "surface":
            sorting_criterion = [detec[2] * detec[3] for detec in detections]
        elif self.nms_mode == "confidence":
            sorting_criterion = confidences
        # else:
        #     sorting_criterion = np.array(confidences) + np.array([detec[2] * detec[3] for detec in detections])
        sorting = np.argsort(sorting_criterion)
        detections = np.array(detections)[sorting]
        
        chosen_ones = [detections[0]]
        for detection in detections[1:]:
            if not check_intersection(chosen_ones, detection, threshold, method=thresholding_method):
                chosen_ones.append(detection)
        return chosen_ones