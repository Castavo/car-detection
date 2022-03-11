from skimage.io import imread
from scripts.utils import run_length_encoding, bounding_boxes_to_mask
from .detection import detection_pipeline
from tqdm import tqdm
from multiprocessing import Pool
import argparse, pickle, os
import csv
import numpy as np


N_PROCESSES = int(os.getenv("SLURM_CPUS_PER_TASK", 1))

parser = argparse.ArgumentParser()
parser.add_argument("--classifier_path", help="The name of the pickle file where the classifier is.")
parser.add_argument("--result_path", help="The name of the pickle file to write the new features to.")
parser.add_argument("--test_dir", help="The name of the directory where the test images are situated.")
parser.add_argument("--n_processes", type=int, help="Number of processes to use.", default=N_PROCESSES)

args = parser.parse_args()

with open(args.classifier_path, "rb") as pickle_file:
    CLASSIFIER = pickle.load(pickle_file)

all_files = [os.path.join(args.test_dir, filename) for filename in os.listdir(args.test_dir)]

def detection_one_image(image_path):
    """This is piping for the multiprocessing module"""
    return (os.path.basename(image_path), *detection_pipeline(image_path, CLASSIFIER))

pool = Pool(args.n_processes)
detections = pool.imap_unordered(
    detection_one_image,
    all_files
)
pool.close()

result = list(tqdm(detections, total=len(all_files)))

os.makedirs(os.path.dirname(args.result_path), exist_ok=True)
with open(args.result_path, "wb") as pickle_file:
    pickle.dump(result, pickle_file)


rows = [["Id", "Predicted"]]
for file_name, bboxes, _, _ in result:
    rle = run_length_encoding(bounding_boxes_to_mask(bboxes, 720, 1280))
    rows.append(['test/' + file_name, rle])

with open(args.result_path.split(".", 1)[0] + ".csv", "w") as dest_file:
    writer = csv.writer(dest_file)
    writer.writerows(rows)