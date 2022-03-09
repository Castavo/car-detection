from sklearn.metrics import confusion_matrix, fbeta_score
import argparse, os, pickle
# from datetime import datetime
from scripts.utils import extract_frames_info, SIMILAR_PREFIXES, remove_small_bb
from .classifier import HOGClassifier

MODELS_PATH = "models/hog"
N_PROCESSES = int(os.getenv("SLURM_CPUS_PER_TASK", 1))

parser = argparse.ArgumentParser()
parser.add_argument("--C", type=float, default=1.0)
parser.add_argument("--gamma", type=float, default=0.01)
parser.add_argument("--tol", type=float, default=0.01)
parser.add_argument("--svm_n_iter", type=int, default=1000)
parser.add_argument("--positive_weight", type=float, default=None)
parser.add_argument("--kernel", default="linear")
parser.add_argument("--goal_shape", type=lambda x: tuple(int(n) for n in x.split(",")), default=(100, 75))
parser.add_argument("--pixels_per_cell", type=lambda x: tuple(int(n) for n in x.split(",")), default=(8, 8))
parser.add_argument("--cells_per_block", type=lambda x: tuple(int(n) for n in x.split(",")), default=(3, 3))
parser.add_argument("--orientations", type=int, default=8)
parser.add_argument("--no_augmentation", action="store_true", help="Do not augment training data by flipping it")
parser.add_argument("--skip_validate", action="store_true", help="Take a part of the data as a validation set or not")
parser.add_argument("--skip_small", action="store_true", help="Ignore small pictures")
parser.add_argument("--model_name", help="The name of the pickle file to write the svm and params to")
parser.add_argument("--skip_metrics", action="store_true", help="Skip computing the metrics")

args = parser.parse_args()

print (vars(args))
frames_info = extract_frames_info("data/train.csv", skip_prefixes=[SIMILAR_PREFIXES[0]] * (not args.skip_validate))

if args.skip_small:
    frames_info = remove_small_bb(frames_info)

hog_params = {
    "orientations": args.orientations, 
    "pixels_per_cell": args.pixels_per_cell, 
    "cells_per_block": args.cells_per_block
}
svm_params = {
    "C": args.C,
    "gamma": args.gamma,
    "kernel": args.kernel,
    "tol": args.tol,
    "max_iter": args.svm_n_iter,
    "class_weight": {0: 1, 1: args.positive_weight}
}

classifier = HOGClassifier(hog_params, svm_params, args.goal_shape)

classifier.train(frames_info, N_PROCESSES, evaluate=not args.skip_metrics, augment=not args.no_augmentation)

os.makedirs(MODELS_PATH, exist_ok=True)
model_name = args.model_name or f"hog_c{args.pixels_per_cell[0]}_s{args.goal_shape[0]}-{args.goal_shape[1]}.pkl"
with open(os.path.join(MODELS_PATH, model_name), "wb") as pickle_file:
    pickle.dump(classifier, pickle_file)

print(f"Model written down in {model_name}")

if not args.skip_validate:
    frames_info = extract_frames_info("data/train.csv", only_prefixes=[SIMILAR_PREFIXES[0]])
    if args.skip_small:
        frames_info = remove_small_bb(frames_info)
    classifier.validate(frames_info, N_PROCESSES)