from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, fbeta_score
import argparse, os, pickle
import numpy as np
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--C", type=float, default=1.0)
parser.add_argument("--degree", default=3, type=int)
parser.add_argument("--gamma", type=float, default=0.01)
parser.add_argument("--kernel", default="rbf")
parser.add_argument("--train_size", default=30_000, type=int)
parser.add_argument("--test_size", default=10_000, type=int)
parser.add_argument("--features_file_path", help="The name of the pickle file where the features vects and labels were writen")
parser.add_argument("--model_name", help="The name of the pickle file to write the svm to")
parser.add_argument("--skip_metrics", action="store_true", help="Skip computing the metrics")

args = parser.parse_args()

print (vars(args))

start_load = datetime.now()
print("Loading features")
with open(args.features_file_path, "rb") as feature_file:
    features, labels = pickle.load(feature_file)
    features = np.array(features)
    labels = np.array(labels)
print(f"Loaded features (took {datetime.now() - start_load})")


start_train = datetime.now()
print("Training model")
mean, std = features.mean(0), np.std(features, 0)
c_features = (features - mean) / std

indices = np.random.choice(len(labels), args.train_size + args.test_size).astype(int)
train_features, train_labels = c_features[indices[:args.train_size]], labels[indices[:args.train_size]]
test_features, test_labels = c_features[indices[args.train_size:]], labels[indices[args.train_size:]]

svm = SVC(C=args.C, degree=args.degree, gamma=args.gamma, kernel=args.kernel, verbose=True)
svm.fit(train_features, train_labels)

print(f"Model trained (took {datetime.now() - start_train})")

if not args.skip_metrics:
    beta_param = .5

    start_evaluate = datetime.now()
    pred_labels = svm.predict(train_features)
    print(f"F-{beta_param} score on the train data: {fbeta_score(train_labels, pred_labels, beta=beta_param)}")

    pred_labels = svm.predict(test_features)
    print(f"F-{beta_param} score on the test data: {fbeta_score(test_labels, pred_labels, beta=beta_param)}")

    print("VN | FP \n FN | VP")
    print(confusion_matrix(test_labels, pred_labels))

    print(f"Evaluation took {datetime.now() - start_evaluate}")


dest_name = args.model_name or "svm_{args.train_size}" + args.reduced * "_reduced" + ".pkl"

with open(dest_name, "wb") as pickle_file:
    pickle.dump(svm, pickle_file)