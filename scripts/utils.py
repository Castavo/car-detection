import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from cv2 import imread
import csv


H, W = 720, 1280

def extract_frames_info(file_path, no_night=False, only_similar=False):
    res = []
    with open(file_path) as csvfile:
        frames_info = csv.reader(csvfile, delimiter=',')
        next(frames_info) # skip the header line
        for frame_info in frames_info:
            if no_night and frame_info[0][6:23] in ["b1c81faa-3df17267", "b1c81faa-c80764c5"]:
                continue
            if only_similar and not frame_info[0][6:23] in ["b1cebfb7-284f5117", "b1c9c847-3bda4659"]:
                continue
            res.append((frame_info[0], annotations_from_csv(frame_info[1])))
    return res

def read_frame(df_annotation, frame):
    """Read frames and create integer frame_id-s"""
    file_path = df_annotation[df_annotation.index == frame]['frame_id'].values[0]
    return imread(file_path)

def annotations_from_csv(bb_string):
    if len(bb_string) == 0:
        return []

    bbs = list(map(lambda x : int(x), bb_string.split(' ')))
    return np.array_split(bbs, len(bbs) / 4)

def show_annotation(df_annotation, frame):
    img = read_frame(df_annotation, frame)
    bbs = annotations_for_frame(df_annotation, frame)

    fig, ax = plt.subplots(figsize=(15, 12))

    for x, y, dx, dy in bbs:

        rect = patches.Rectangle((x, y), dx, dy, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    ax.imshow(img)
    ax.set_title('Annotations for frame {}.'.format(frame))

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