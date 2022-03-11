import matplotlib.patches as patches
import matplotlib.pyplot as plt
from skimage.io import imread


def show_bbs(ax, bbs, last_green=False, confidences=None):
    for i, (x, y, dx, dy) in enumerate(bbs):
        ax.add_patch(
            patches.Rectangle(
                (x, y), dx, dy, 
                edgecolor='g' if (last_green and i == len(bbs) - 1)  else 'r', 
                facecolor='none'
            )
        )
        if confidences is not None:
            ax.text(
                x, y, str(confidences[i]), 
                backgroundcolor="r", color="w", 
                horizontalalignment="left", verticalalignment="bottom", 
                fontsize=8, fontweight="bold"
            )

def show_image_with_bbs(image_path, bbs, last_green=False, confidences=None):
    image = imread(image_path)
    _, ax = plt.subplots(figsize=(15, 8))
    ax.imshow(image)
    show_bbs(ax, bbs, last_green, confidences)