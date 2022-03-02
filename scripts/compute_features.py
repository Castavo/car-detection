import pandas as pd
import numpy as np
from cv2 import SIFT_create
from sklearn.cluster import KMeans
from tqdm import tqdm
import os
import pickle
from utils import read_frame, annotations_for_frame

N_FEATURES = 500_000
FEATURES_PICKLE_NAME = "data/features.pkl"

def collect_labeled_features(df_annotation, sift):
    """ 
    Returns SIFT features for all images labeled as "part of a car" or not
    The labeling is done thanks to the keypoints and bounding boxes
    """
    features, labels = [], []

    for frame in tqdm(range(1, len(df_annotation))):
        image = read_frame(df_annotation, frame)
        bbs = annotations_for_frame(df_annotation, frame)
        kp, f = sift.detectAndCompute(image, None)

        features += f.tolist()
        labels += label_keypoints(kp, bbs).tolist()

    return features, labels

def label_keypoints(keypoints, bounding_boxes):
    if len(bounding_boxes) == 0:
        print("Well well well")
        return np.zeros(len(keypoints), np.integer)

    coords = np.array([kp.pt for kp in keypoints])

    low_bbs = np.array([bb[0:2] for bb in bounding_boxes]) # shape = (M, 2)
    high_bbs = low_bbs + np.array([bb[2:] for bb in bounding_boxes])

    is_coord_good = (low_bbs[np.newaxis, ...] <= coords[:, np.newaxis, :]) & (coords[:, np.newaxis, :] <= high_bbs[np.newaxis, ...]) # (N, M, 2)

    is_kp_good = (is_coord_good[:, :, 0] & is_coord_good[:, :, 1]).any(1)

    return is_kp_good.astype(int)

def reduce_features(features, labels, n_features):
    print("Reducing features")
    kmeans = KMeans(n_features, verbose=2)
    afiliations = kmeans.fit_predict(features)

    print("Computing new labels")
    centroid_labels = []
    count_bofbof = 0
    for i in range(n_features):
        new_label = labels[afiliations == i].mean()
        if .4 < new_label <.6:
            count_bofbof += 1
        centroid_labels.append(int(new_label >= .6))
    print(f"Il y a quand même eu {count_bofbof} sur {n_features}")
    return kmeans.cluster_centers_, centroid_labels


if __name__ == "__main__":
    df_ground_truth = pd.read_csv('data/train.csv')

    if not os.path.exists(FEATURES_PICKLE_NAME):
        print(
            "SIFT features were not already computed.\n" 
            f"Computing them and writing them into {FEATURES_PICKLE_NAME}"
        )
        sift = SIFT_create()
        features, labels = collect_labeled_features(df_ground_truth, sift)
        if len(features) > N_FEATURES:
            print(
                f"We gathered {len(features)} feature, which is to many to store."
                "Compressing them using KMeans"
            )
            features, labels = reduce_features(features, labels, N_FEATURES)

        with open(FEATURES_PICKLE_NAME, "wb") as pickle_file:
            pickle.dump((features, labels), pickle_file)
    else:
        print(
            "SIFT features were already computed.\n" 
            f"Loading them from {FEATURES_PICKLE_NAME}"
        )
        with open(FEATURES_PICKLE_NAME, "rb") as pickle_file:
            features, labels = pickle.load(pickle_file)
    print("Done")
