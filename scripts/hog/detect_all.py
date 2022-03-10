from skimage.io import imread
from scripts.utils import extract_frames_info, run_length_encoding, bounding_boxes_to_mask
from .detector import Detector
from tqdm import tqdm
from multiprocessing import Pool
import argparse, pickle, os
import csv


N_PROCESSES = int(os.getenv("SLURM_CPUS_PER_TASK", 1))

WINDOW_WIDTHS, WINDOW_RATIOS = [64, 128, 256, 400], [1.0, .8, .5]
WINDOW_SHAPES = [(width, int(width*ratio)) for width in WINDOW_WIDTHS for ratio in WINDOW_RATIOS]

parser = argparse.ArgumentParser()
parser.add_argument("--classifier_path", help="The name of the pickle file where the classifier is.")
parser.add_argument("--result_path", help="The name of the pickle file to write the new features to.")
parser.add_argument("--test_dir", help="The name of the directory where the test images are situated.")

args = parser.parse_args()

with open(args.classifier_path, "rb") as pickle_file:
    classifier = pickle.load(pickle_file)

detector = Detector(classifier)

def detect_one_image(image_path):
    image = imread(image_path)
    rough_detections = detector.detect(image, 10, [64, 128, 256, 400], [.8, 0.5, 1.0], verbose=False)
    detections = detector.nms(rough_detections)
    return os.path.basename(image_path), detections

all_files = [os.path.join(args.test_dir, filename) for filename in os.listdir(args.test_dir)]

pool = Pool(N_PROCESSES)
detections = pool.imap_unordered(
    detect_one_image,
    all_files
)
pool.close()

result = list(tqdm(detections, total=len(all_files)))

with open(args.result_path, "wb") as pickle_file:
    pickle.dump(result, pickle_file)


rows = [["Id", "Predicted"]]
for file_name, bboxes in detections:
    rle = run_length_encoding(bounding_boxes_to_mask(bboxes, 720, 1280))
    rows.append(['test/' + file_name, rle])

with open(args.result_path.split(".", 1)[0] + ".csv", "w") as dest_file:
    writer = csv.writer(dest_file)
    writer.writerows(rows)