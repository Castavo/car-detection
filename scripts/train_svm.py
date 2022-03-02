from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, fbeta_score
import argparse
import pickle
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--C", type=float, default=1.0)
parser.add_argument("--degree", default=3, type=int)
parser.add_argument("--gamma", type=float, default=0.01)
parser.add_argument("--kernel", default="rbf")
parser.add_argument("--train_size", default=30_000, type=int)
parser.add_argument("--test_size", default=10_000, type=int)
parser.add_argument("--reduced", action="store_true")

args = parser.parse_args()

print (vars(args))

print("Loading features")
with open("data/" + ("reduced_features.pkl" if args.reduced else "small_features.pkl"), "rb") as feature_file:
    features, labels = pickle.load(feature_file)
    features = np.array(features)
    labels = np.array(labels)
print("Loaded features")


print("Training model")
mean, std = features.mean(0), np.std(features, 0)
c_features = (features - mean) / std

indices = np.random.choice(len(labels), args.train_size + args.test_size).astype(int)
train_features, train_labels = c_features[indices[:args.train_size]], labels[indices[:args.train_size]]
test_features, test_labels = c_features[indices[args.train_size:]], labels[indices[args.train_size:]]

svm = SVC(C=args.C, degree=args.degree, gamma=args.gamma, kernel=args.kernel, verbose=True)
svm.fit(train_features, train_labels)

beta_param = .5

pred_labels = svm.predict(train_features)
print(f"F-{beta_param} score on the train data: {fbeta_score(train_labels, pred_labels, beta=beta_param)}")

pred_labels = svm.predict(test_features)
print(f"Precision on the test data: {fbeta_score(test_labels, pred_labels, beta=beta_param)}")

print("VN | FP \n FN | VP")
print(confusion_matrix(test_labels, pred_labels))