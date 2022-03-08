from sklearn.metrics import confusion_matrix, fbeta_score
import argparse, os, pickle
# from datetime import datetime
from scripts.utils import extract_frames_info
from .classifier import HOGClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--C", type=float, default=1.0)
parser.add_argument("--gamma", type=float, default=0.01)
parser.add_argument("--kernel", default="linear")
parser.add_argument("--validate", action="store_true", help="Take a part of the data as a validation set")
parser.add_argument("--skip_small", action="store_true", help="Ignore small pictures")
parser.add_argument("--model_name", help="The name of the pickle file to write the svm and params to")
parser.add_argument("--skip_metrics", action="store_true", help="Skip computing the metrics")

args = parser.parse_args()

print (vars(args))

frames_info = extract_frames_info("data/train.csv")

classifier = HOGClassifier(C=args.C, gamma=args.gamma, kernel=args.kernel)

classifier.features_labels(frames_info[:100], 1)
