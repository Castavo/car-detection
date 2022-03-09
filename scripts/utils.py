import numpy as np
import csv, os

NIGHT_PREFIXES = ["b1c81faa-3df17267", "b1c81faa-c80764c5"]
SIMILAR_PREFIXES = ["b1cebfb7-284f5117", "b1c9c847-3bda4659"]

H, W = 720, 1280

def extract_frames_info(file_path, data_path="data", only_prefixes=None, skip_prefixes=None):
    res = []
    with open(file_path) as csvfile:
        frames_info = csv.reader(csvfile, delimiter=',')
        next(frames_info) # skip the header line
        for frame_info in frames_info:
            if skip_prefixes and frame_info[0][6:23] in skip_prefixes:
                continue
            if only_prefixes and not frame_info[0][6:23] in only_prefixes:
                continue
            res.append(
                (os.path.join(data_path, frame_info[0]), annotations_from_string(frame_info[1]))
            )
    return res

def annotations_from_string(bb_string):
    if len(bb_string) == 0:
        return []

    bbs = list(map(int, bb_string.split(' ')))
    return np.array_split(bbs, len(bbs) / 4)

def label_keypoints(keypoints, bounding_boxes):
    """Returns an array saying for each kp if it is in the bb or not"""
    if len(bounding_boxes) == 0:
        return np.zeros(len(keypoints), np.int64)

    if type(keypoints) != np.ndarray:
        coords = np.array([kp.pt for kp in keypoints])
    else:
        coords = keypoints

    low_bbs = np.array([bb[0:2] for bb in bounding_boxes]) # shape = (M, 2)
    high_bbs = low_bbs + np.array([bb[2:] for bb in bounding_boxes])

    is_coord_good = (low_bbs[np.newaxis, ...] <= coords[:, np.newaxis, :]) & (coords[:, np.newaxis, :] <= high_bbs[np.newaxis, ...]) # (N, M, 2)

    is_kp_good = (is_coord_good[:, :, 0] & is_coord_good[:, :, 1]).any(1)

    return is_kp_good.astype(int)

def bounding_boxes_to_mask(bounding_boxes, H, W):
    
    """
    Converts set of bounding boxes to a binary mask
    """

    mask = np.zeros((H, W))
    for x, y, dx, dy in bounding_boxes:
        mask[y:y+dy, x:x+dx] = 1

    return mask

def run_length_encoding(mask):
    """
    Produces run length encoding for a given binary mask
    """
    
    # find mask non-zeros in flattened representation
    non_zeros = np.nonzero(mask.flatten())[0]
    padded = np.pad(non_zeros, pad_width=1, mode='edge')
    
    # find start and end points of non-zeros runs
    limits = (padded[1:] - padded[:-1]) != 1
    starts = non_zeros[limits[:-1]]
    ends = non_zeros[limits[1:]]
    lengths = ends - starts + 1

    return ' '.join(['%d %d' % (s, l) for s, l in zip(starts, lengths)])

def remove_small_bb(frames_info, min_size=64):
    return [
        (path, [bb for bb  in bbs if min(bb[2], bb[2]) > min_size]) 
        for path, bbs in frames_info
    ]

if __name__ == "__main__":
    res = [ar.tolist() for ar in annotations_from_string("0 0 100 100 20 50 125 165")]
    assert res == [[0, 0, 100, 100], [20, 50, 125, 165]]
    print("All good")