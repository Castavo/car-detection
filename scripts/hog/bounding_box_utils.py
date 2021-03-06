import numpy as np
from random import randint

def intersection_area(bbox1, bbox2):
    """bbox: (x, y, deltax, deltay)"""
    return np.prod([
        max(0, min(bbox1[i] + bbox1[i+2], bbox2[i] + bbox2[i+2]) - max(bbox1[i], bbox2[i]))
        for i in [0, 1]
    ])

def check_intersection(bounding_boxes, bb_candidate, threshold=.25, method="IOU"):
    """check wether the candidate intersects with another bouding box 
    for at least <threshold> of IOU or min surface coverage"""
    for bbox in bounding_boxes:
        if method == "IOU":
            ref = (bbox[2] * bbox[3] + bb_candidate[2] * bb_candidate[3]) / 2
        else:
            ref = min(bbox[2] * bbox[3], bb_candidate[2] * bb_candidate[3])

        if intersection_area(bbox, bb_candidate) / ref > threshold:
            return True
    return False

def random_position_in_image(image_shape, window_shape):
    return (
        randint(0, image_shape[0] - window_shape[0]), 
        randint(0, image_shape[1] - window_shape[1])
    )

def find_free_window(image_shape, bounding_boxes, window_shape, at_random=True):
    """Returns a position where we could have a free window in our image"""
    if at_random:
        window_pos = random_position_in_image(image_shape, window_shape)
        i = 0
        while check_intersection(bounding_boxes, (*window_pos, *window_shape)):
            window_pos = random_position_in_image(image_shape, window_shape)
            i += 1
            if i > 1000:
                # Window is just too big
                return None
        return (*window_pos, *window_shape)
        
def place_all_windows(image_shape, bounding_boxes, window_shapes, ignore_other_windows=False):
    new_bboxes = []
    for window_shape in window_shapes:
        new_bbox = find_free_window(
            image_shape, 
            bounding_boxes + new_bboxes * (not ignore_other_windows), 
            window_shape
        )
        if new_bbox:
            new_bboxes.append(new_bbox)
    return new_bboxes

if __name__ == "__main__":
    assert intersection_area([0, 0, 100, 200], [50, 100, 100, 200]) == 5000
    assert intersection_area([0, 200, 100, 200], [50, 100, 100, 200]) == 5000
    assert intersection_area([50, 100, 100, 200], [0, 0, 100, 200]) == 5000
    assert intersection_area([0, 0, 100, 200], [25, 25, 50, 150]) == 50*150
    print("All good")