from cv2 import SIFT_create, imread
from sklearn.decomposition import PCA
from tqdm import tqdm
from multiprocessing import Pool
from datetime import datetime
from utils import extract_frames_info
import os, pickle, argparse
import numpy as np

FEATURES_PICKLE_NAME = "data/features.pkl"

def collect_labeled_vectors(frames_info, stride=10, n_processes=1):
    """ 
        Returns SIFT feature vectors for all images, labeled as "part of a car" or not
        The labeling is done thanks to the keypoints and bounding boxes
        We used a stride because these are images coming from a video, and consequent images will contain similar SIFT vectors
    """
    sift = SIFT_create()
    
    def labeled_vectors_for_frame(frame_path, bbs):
        image = imread(os.path.join("data", frame_path))
        kp, f = sift.detectAndCompute(image, None)
        return f, label_keypoints(kp, bbs)

    vectors, labels = [], []
    if n_processes != 1:
        pool = Pool(n_processes)
        vects_labels = pool.starmap(
            labeled_vectors_for_frame,
            frames_info[::stride],
        )

        pool.close()

        # gather vectors and labels in one list, we 
        for feature_vects, associated_labels in tqdm(vects_labels):
            vectors += feature_vects.tolist()
            labels += associated_labels.tolist()
    else:
        for frame_path, bb_string in tqdm(frames_info[::stride]):
            feature_vects, associated_labels = labeled_vectors_for_frame(frame_path, bb_string)
            vectors += feature_vects.tolist()
            labels += associated_labels.tolist()
            
    return vectors, labels

def label_keypoints(keypoints, bounding_boxes):
    """Returns an array saying for each kp if it is in the bb or not"""
    if len(bounding_boxes) == 0:
        return np.zeros(len(keypoints), np.int64)

    coords = np.array([kp.pt for kp in keypoints])

    low_bbs = np.array([bb[0:2] for bb in bounding_boxes]) # shape = (M, 2)
    high_bbs = low_bbs + np.array([bb[2:] for bb in bounding_boxes])

    is_coord_good = (low_bbs[np.newaxis, ...] <= coords[:, np.newaxis, :]) & (coords[:, np.newaxis, :] <= high_bbs[np.newaxis, ...]) # (N, M, 2)

    is_kp_good = (is_coord_good[:, :, 0] & is_coord_good[:, :, 1]).any(1)

    return is_kp_good.astype(int)

def reduce_n_features(feature_vects, n_features):
    """Reduce the dimensionnality of the features vectors"""
    print("Reducing features")
    pca = PCA(n_features)
    reduced_vectors = pca.fit_transform(feature_vects)
    return reduced_vectors, pca


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--n_features", type=int, default=20)
    parser.add_argument("--n_processes", type=int, default=1)
    parser.add_argument("--features_filename", default=FEATURES_PICKLE_NAME)
    parser.add_argument("--pca_filename", help="The name of the pickle file to write the pca to", default="models/pca.pkl")
    parser.add_argument("--force", action="store_true", help="force recomputing if result file already exists")

    args = parser.parse_args()

    if os.path.exists(args.features_filename) and not args.force:
        print("Feature vectors file already exists, quitting.")
        exit()

    print(
        f"Computing SIFT features them and writing them into {args.features_filename}"
    )
    
    frames_info = extract_frames_info('data/train.csv')
    start_sift = datetime.now()
    feature_vects, labels = collect_labeled_vectors(frames_info, args.stride, args.n_processes)
    print(f"We gathered {len(labels)} feature vectors using a stride of {args.stride}")
    print(f"That took {datetime.now() - start_sift}")

    start_pca = datetime.now()
    feature_vects, pca = reduce_n_features(feature_vects, args.n_features)
    print(
        f"Reduced vectors account for {pca.explained_variance_ratio_.sum()}"
        f"of the ratio, with dimension {args.n_features}"
    )
    print(f"That took {datetime.now() - start_pca}")

    os.makedirs(os.path.dirname(args.pca_filename), exist_ok=True)
    with open(args.pca_filename, "wb") as pickle_file:
        pickle.dump(pca, pickle_file)

    with open(args.features_filename, "wb") as pickle_file:
        pickle.dump((feature_vects, labels), pickle_file)

    print("All files writen down")