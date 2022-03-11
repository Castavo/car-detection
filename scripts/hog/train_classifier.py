import argparse, os, pickle
from scripts.utils import extract_frames_info, SIMILAR_PREFIXES, NIGHT_PREFIXES, remove_small_bb
from .classifier import HOGClassifier

MODELS_PATH = "models/hog"
N_PROCESSES = int(os.getenv("SLURM_CPUS_PER_TASK", 1))

parser = argparse.ArgumentParser()
parser.add_argument("--C", type=float, default=0.01)
parser.add_argument("--gamma", type=float, default=0.01)
parser.add_argument("--tol", type=float, default=0.01)
parser.add_argument("--svm_n_iter", type=int, default=3000)
parser.add_argument("--positive_weight", type=float, default=.2)
parser.add_argument("--kernel", default="linear")
parser.add_argument("--goal_shape", type=lambda x: tuple(int(n) for n in x.split(",")), default=(64, 64))
parser.add_argument("--pixels_per_cell", type=lambda x: tuple(int(n) for n in x.split(",")), default=(8, 8))
parser.add_argument("--cells_per_block", type=lambda x: tuple(int(n) for n in x.split(",")), default=(3, 3))
parser.add_argument("--orientations", type=int, default=8)
parser.add_argument("--no_augmentation", action="store_true", help="Do not augment training data by flipping it")
parser.add_argument("--hard_examples_path", help="The path to the hard examples features")
parser.add_argument("--skip_validate", action="store_true", help="Take a part of the data as a validation set or not")
parser.add_argument("--no_night", action="store_true", help="Do not train on night time videos")
parser.add_argument("--skip_small", action="store_true", help="Ignore small pictures")
parser.add_argument("--model_filename", help="The name of the pickle file to write the svm and params to")
parser.add_argument("--skip_metrics", action="store_true", help="Skip computing the metrics")
parser.add_argument("--n_processes", type=int, help="Number of processes to use.", default=N_PROCESSES)

args = parser.parse_args()
print(vars(args))

frames_info = extract_frames_info(
    "data/train.csv", 
    skip_prefixes=[SIMILAR_PREFIXES[0]] * (not args.skip_validate) + (not args.no_night) * NIGHT_PREFIXES)

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

if args.hard_examples_path:
    with open(args.hard_examples_path, "rb") as pickle_file:
        hard_features = pickle.load(pickle_file)
else:
    hard_features = None

classifier = HOGClassifier(hog_params, svm_params, args.goal_shape)
classifier.train(
    frames_info,
    args.n_processes,
    evaluate=not args.skip_metrics, 
    augment=not args.no_augmentation,
    hard_examples=hard_features
)


os.makedirs(MODELS_PATH, exist_ok=True)
model_name = args.model_filename or f"hog_c{args.pixels_per_cell[0]}_s{args.goal_shape[0]}-{args.goal_shape[1]}.pkl"
with open(os.path.join(MODELS_PATH, model_name), "wb") as pickle_file:
    pickle.dump(classifier, pickle_file)

print(f"Model written down in {model_name}")

if not args.skip_validate:
    frames_info = extract_frames_info("data/train.csv", only_prefixes=[SIMILAR_PREFIXES[0]])
    if args.skip_small:
        frames_info = remove_small_bb(frames_info)
    classifier.validate(frames_info, args.n_processes)