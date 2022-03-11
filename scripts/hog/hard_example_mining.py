from skimage.io import imread
from .bounding_box_utils import find_free_window
from scripts.utils import extract_frames_info
from random import choice
from tqdm import tqdm
from multiprocessing import Pool
import argparse, pickle, os


N_PROCESSES = int(os.getenv("SLURM_CPUS_PER_TASK", 1))

WINDOW_WIDTHS, WINDOW_RATIOS = [64, 128, 256, 400, 512], [1.0, .8, .5]
WINDOW_SHAPES = [(width, int(width*ratio)) for width in WINDOW_WIDTHS for ratio in WINDOW_RATIOS]

parser = argparse.ArgumentParser()
parser.add_argument("--classifier_path", help="The name of the pickle file where the classifier is.")
parser.add_argument("--result_path", help="The name of the pickle file to write the new features to.")
parser.add_argument(
    "--n_examples_per_image", 
    help="The name of the pickle file to write the svm and params to.", 
    type=int, default=20
)
parser.add_argument("--n_processes", type=int, help="Number of processes to use.", default=N_PROCESSES)

args = parser.parse_args()

with open(args.classifier_path, "rb") as pickle_file:
    CLASSIFIER = pickle.load(pickle_file)

def mine_hard_example(frame_info):
    image_path, bboxes = frame_info
    examples = []
    image = imread(image_path)
    while len(examples) < args.n_examples_per_image:
        window = find_free_window((image.shape[1], image.shape[0]), bboxes, choice(WINDOW_SHAPES))
        if window is None:
            continue
        sub_image = image[window[1]: window[1]+window[3], window[0]: window[0]+window[2]]
        label, feature_vect = CLASSIFIER.predict(sub_image, return_feature=True)
        if label:
            examples.append(feature_vect)
    return examples


frames_info = extract_frames_info("data/train.csv")

pool = Pool(args.n_processes)
hard_examples = pool.imap_unordered(
    mine_hard_example,
    frames_info
)

pool.close()

result = []
for example_list in tqdm(hard_examples, total=len(frames_info)):
    result += example_list

os.makedirs(os.path.dirname(args.result_path), exist_ok=True)
with open(args.result_path, "wb") as pickle_file:
    pickle.dump(result, pickle_file)