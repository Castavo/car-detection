from cv2 import imread
from skimage.feature import hog
from skimage.transform import resize
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import f1_score, confusion_matrix
from multiprocessing import Pool
from tqdm.auto import tqdm
from random import shuffle
from .bounding_box_utils import place_all_windows
import numpy as np

def compute_features_image(file_path, bounding_boxes, windows_to_place, goal_shape, hog_params):
    labels = [1] * len(bounding_boxes) + [0] * len(windows_to_place)
    image = imread(file_path)
    bounding_boxes = place_all_windows((image.shape[1], image.shape[0]), bounding_boxes, windows_to_place)
    features = []
    for bb in bounding_boxes:
        sub_image = image[bb[1]: bb[1] + bb[3], bb[0]: bb[0] + bb[2]]
        sub_image = resize(sub_image, goal_shape)
        features.append(hog(sub_image, **hog_params))
    return features, labels


class HOGClassifier:
    def __init__(
        self, 
        orientations=16, 
        pixels_per_cell=(8, 8), 
        cells_per_block=(4, 4), 
        C=1.0, gamma=0.01, kernel="linear",
        goal_shape=(100, 150)
    ):
        """Careful, goal_shape is reversed (compared to the intuition)"""
        self.goal_shape = goal_shape
        self.hog_params = {
            "orientations": orientations, 
            "pixels_per_cell": pixels_per_cell, 
            "cells_per_block": cells_per_block
        }
        self.svm_params = {
            "C": C,
            "gamma": gamma,
            "kernel": kernel
        }
        # Will be computed before feeding the SVM
        self.mean = None 
        self.std = None
        self.svm = None
    
    def features_labels(self, frames_info, n_processes=10):
        """Returns a list of feature vectors with a list of labels in the same order
        frames_info list of tuples: filename, bounding_boxes"""
        # print(frames_info[0])
        all_bb_shapes = sum([[bb[2:] for bb in info[1]] for info in frames_info], []) * 2
        shuffle(all_bb_shapes)
        negatives_window_shapes = np.array_split(all_bb_shapes, len(frames_info))

        global extract_features
        def extract_features(one_arg):
            return compute_features_image(one_arg[0][0], one_arg[0][1], one_arg[1], self.goal_shape, self.hog_params)

        if n_processes != 1:
            pool = Pool(n_processes)
            features_labels = pool.imap_unordered(
                extract_features, 
                zip(frames_info, negatives_window_shapes)
            )
            pool.close()
        else:
            features_labels = [
                extract_features(one_arg) 
                for one_arg in tqdm(
                    zip(frames_info, negatives_window_shapes),total=len(frames_info)
                )
            ]
        features, labels = [], []
        for feature_sublist, label_sublit in tqdm(features_labels, total=len(frames_info)):
            features += feature_sublist
            labels += label_sublit
        return features, labels


    def train(self, frames_info, n_processes=10, verbose=2, evaluate=True):
        if verbose: print("Computing features")
        features, labels = self.features_labels(frames_info, n_processes)

        if verbose: print("Done.\nTraining SVM")
        self.mean = np.mean(features, axis=0)
        self.std = np.std(features, axis=0)

        c_features = (features - self.mean) / self.std

        if self.svm_params["kernel"] == "linear":
            self.svm_params.pop("kernel")
            self.svm_params.pop("gamma")
            self.svm = LinearSVC(
                fit_intercept=False, 
                dual=len(labels) > len(features[0]), 
                verbose=verbose,
                **self.svm_params, 
            )
        else:
            self.svm = SVC(
                verbose=verbose>0,
                **self.svm_params, 
            )
        self.svm.fit(c_features, labels)
        if verbose: print("Done.")

        if evaluate:
            if verbose: print("Evaluating SVM")
            pred_labels = self.svm.predict(c_features)
            print(confusion_matrix(labels, pred_labels))
            print(f"F1-score over the train data: {f1_score(labels, pred_labels)}")

