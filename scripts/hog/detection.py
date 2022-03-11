from skimage.io import imread
from tqdm.auto import tqdm
import numpy as np

from .bounding_box_utils import check_intersection

WINDOW_WIDTHS, WINDOW_RATIOS = np.linspace(64, 300, 4, dtype=int), [1.0, .75, .5]

def detect(classifier, image, stride, window_widths, window_ratios, verbose=True, height_range=None):
    """
    Uses the sliding window method to detect cars thanks to the classifier
    height_range helps the detection by telling where not to look 
    (there are no cars in the skies nor on the dashboard of the car in the video)
    """
    detections = []
    decisions = []

    H, W = image.shape[:2]
    height_range = height_range or [0, H]
    if verbose: progress = tqdm(total=len(window_widths)*len(window_ratios))
    for width in window_widths:
        for ratio in window_ratios:
            if verbose: progress.update()
            window_shape = (width, int(width * ratio))
            for i in range(height_range[0], height_range[1] - window_shape[1], stride):
                for j in range(0, W - window_shape[0], stride):
                    sub_image = image[i: i + window_shape[1], j: j + window_shape[1]]
                    decision = classifier.predict(sub_image, return_decision=True)[0]
                    if decision > 0:
                        detections.append((j, i, *window_shape))
                        decisions.append(decision)
    if verbose: progress.close()
    return detections, decisions

def nms(detections, mode="surface", threshold=.75, thresholding_method="min", confidences=None):
    """Performs non maximum suppression on the given detections"""
    if len(detections) == 0:
        return []
    if mode == "surface":
        sorting_criterion = [detec[2] * detec[3] for detec in detections]
    elif mode == "confidence":
        sorting_criterion = confidences
    else:
        raise NotImplementedError
    sorting = np.argsort(sorting_criterion)
    detections = np.array(detections)[sorting]
    
    chosen_ones = [detections[0]]
    for detection in detections[1:]:
        if not check_intersection(chosen_ones, detection, threshold, method=thresholding_method):
            chosen_ones.append(detection)
    return chosen_ones


def detection_pipeline(image_path, classifier):
    """Performs car detection on the given image"""
    image = imread(image_path)
    rough_detections, decisions = detect(classifier, image, 5, WINDOW_WIDTHS, WINDOW_RATIOS, verbose=False, height_range=[100, 500])
    best_idx = np.array(decisions) > .2
    best_detections = np.array(rough_detections)[best_idx]
    best_decisions = np.array(decisions)[best_idx]
    detections = nms(best_detections, "confidence", .5, "min", best_decisions)
    return detections, rough_detections, decisions