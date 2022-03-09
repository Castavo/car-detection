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

def compute_features_image(file_path, bounding_boxes, windows_to_place, goal_shape, hog_params, augment=True):
    image = imread(file_path)
    placed_windows = place_all_windows((image.shape[1], image.shape[0]), bounding_boxes, windows_to_place, True)
    labels = [1] * len(bounding_boxes) * (1 + augment) + [0] * len(placed_windows) * (1 + augment)
    features = []
    for bb in bounding_boxes + placed_windows:
        sub_image = image[bb[1]: bb[1] + bb[3], bb[0]: bb[0] + bb[2]]
        sub_image = resize(sub_image, goal_shape)
        features.append(hog(sub_image, **hog_params, channel_axis=-1))
        if augment:
            sub_image = sub_image[:, ::-1]
            features.append(hog(sub_image, **hog_params, channel_axis=-1))
    return features, labels


class HOGClassifier:
    def __init__(
        self, hog_params, svm_params, goal_shape
    ):
        """Careful, goal_shape is reversed (compared to the intuition)"""
        self.goal_shape = goal_shape
        self.hog_params = hog_params or {}
        self.svm_params = svm_params or {}
        # Will be computed before feeding the SVM
        self.svm = None
    
    def features_labels(self, frames_info, n_processes=10, augment=True, n_negatives=5):
        """Returns a list of feature vectors with a list of labels in the same order
        frames_info list of tuples: filename, bounding_boxes"""
        # print(frames_info[0])
        all_bb_shapes = sum([[bb[2:] for bb in info[1]] for info in frames_info], []) * n_negatives
        shuffle(all_bb_shapes)
        negatives_window_shapes = np.array_split(all_bb_shapes, len(frames_info))

        global extract_features
        def extract_features(one_arg):
            return compute_features_image(one_arg[0][0], one_arg[0][1], one_arg[1], self.goal_shape, self.hog_params, augment)

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

    def train(self, frames_info, n_processes=10, verbose=2, evaluate=True, augment=True):
        if verbose: print("Computing features")
        features, labels = self.features_labels(frames_info, n_processes, augment)

        if verbose: print(f"Done. We have a total of {len(features)} features of length {len(features[0])}\nTraining SVM")

        if self.svm_params["kernel"] == "linear":
            self.svm_params.pop("kernel")
            self.svm_params.pop("gamma")
            self.svm = LinearSVC(
                fit_intercept=True, 
                dual=len(labels) > len(features[0]), 
                verbose=verbose,
                **self.svm_params, 
            )
        else:
            self.svm = SVC(
                verbose=verbose>0,
                **self.svm_params, 
            )
        self.svm.fit(features, labels)
        if verbose: print("Done.")

        if evaluate:
            if verbose: print("Evaluating SVM")
            pred_labels = self.svm.predict(features)
            print(confusion_matrix(labels, pred_labels))
            print(f"F1-score over the train data: {f1_score(labels, pred_labels)}")

    def predict(self, image, return_feature=False):
        resized = resize(image, self.goal_shape)
        feature_vect = hog(resized, **self.hog_params, channel_axis=-1)
        label = self.svm.predict([feature_vect])
        if return_feature:
            return label, feature_vect
        else:
            return label

    def mass_predict(self, images, n_processes):
        global compute_features
        def compute_features(image):
            resized = resize(image, self.goal_shape)
            return hog(resized, **self.hog_params, channel_axis=-1)
        pool = Pool(n_processes)
        features = pool.imap_unordered(
            extract_features, 
            images
        )
        pool.close()
        return self.svm.predict(list(tqdm(features)))
    
    def validate(self, frames_info, n_processes):
        features, labels = self.features_labels(frames_info, n_processes, False, 5)
        pred_labels = self.svm.predict(features)
        print("Validation: ")
        print(confusion_matrix(labels, pred_labels))
        print(f"F1-score: {f1_score(labels, pred_labels)}")

    def fit(self, features, labels, svm_params):
        
        self.mean = np.mean(features, axis=0)
        self.std = np.std(features, axis=0)

        c_features = (features - self.mean) / self.std
        self.svm = LinearSVC(
            fit_intercept=False, 
            dual=len(labels) > len(features[0]), 
            verbose=0,
            **svm_params, 
        )
        