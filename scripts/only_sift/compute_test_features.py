from cv2 import SIFT_create, imread
from sklearn.decomposition import PCA
from tqdm import tqdm
import os, pickle, argparse
import numpy as np


TEST_DIR = "data/test"

def collect_vectors(frame_filenames):
    sift = SIFT_create()
    feature_vects, kps = [], []
    for frame_filename in tqdm(frame_filenames):
        image = imread(os.path.join(TEST_DIR, frame_filename))
        kp, f = sift.detectAndCompute(image, None)
        feature_vects.append(f.tolist())
        kps.append(kp)
    return feature_vects, kps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_features", type=int, default=20)
    parser.add_argument("--features_filename", default=None)
    parser.add_argument("--pca_filename", help="The name of the pickle file where the pca is", default=None)
    parser.add_argument("--force", action="store_true", help="force recomputing if result file already exists")

    args = parser.parse_args()

    if os.path.exists(args.features_filename) and not args.force:
        print("Feature vectors file already exists, quitting.")
        exit()

    print(
        f"Computing SIFT features and writing them into {args.features_filename}"
    )
    test_frames_filenames = os.listdir(TEST_DIR)
    feature_vects, keypoints = collect_vectors(test_frames_filenames)

    all_vectors = np.array(sum(feature_vects, []))
    print(f"We gathered {all_vectors.shape} feature vectors")
    # print(all_vectors[:100])

    if os.path.exists(args.pca_filename):
        with open(args.pca_filename, "rb") as pickle_file:
            pca = pickle.load(pickle_file)
    else:
        pca = PCA(args.n_features).fit(all_vectors)
        with open(args.pca_filename, "wb") as pickle_file:
            pickle.dump(pca, pickle_file)

    reduced_vects = [pca.transform(vects) for vects in feature_vects]

    with open(args.features_filename, "wb") as pickle_file:
        pickle.dump((reduced_vects, [[kp.pt for kp in kps] for kps in keypoints]), pickle_file)

    print("All done")