from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, make_scorer, fbeta_score
import argparse
import pickle
import numpy as np
import json

parser = argparse.ArgumentParser()
parser.add_argument("--train_size", default=30_000, type=int)
parser.add_argument("--reduced", action="store_true")
parser.add_argument("--n_folds", type=int, default=3)

args = parser.parse_args()

print(f"Args: {{'train_size': {args.train_size}, 'reduced': {args.reduced}}}")

print("Loading features")
with open("data/" + ("reduced_features.pkl" if args.reduced else "small_features.pkl"), "rb") as feature_file:
    features, labels = pickle.load(feature_file)
    features = np.array(features)
    labels = np.array(labels)
print("Loaded features")

tuned_parameters = [
    {"kernel": ["rbf"], "gamma": np.logspace(-2, 1, 7), "C": np.logspace(-1, 2, 7)},
    {"kernel": ["poly"], "degree": [3, 4, 5, 6, 7, 8], "C": np.logspace(-1, 2, 7)},
]

mean, std = features.mean(0), np.std(features, 0)
c_features = (features - mean) / std
indices = np.random.choice(len(labels), args.train_size).astype(int)
train_features, train_labels = c_features[indices], labels[indices]

beta_param = .5
scorer = make_scorer(fbeta_score, beta=beta_param, zero_division=0)
print(f"Using F-{beta_param} score")

grid = GridSearchCV(
    SVC(), 
    tuned_parameters, 
    scoring=scorer, 
    n_jobs=-1, 
    cv=args.n_folds, 
    verbose=2
)
grid.fit(train_features, train_labels)

print("All scores :")
means = grid.cv_results_["mean_test_score"]
stds = grid.cv_results_["std_test_score"]
results = {}
for mean, std, params in zip(means, stds, grid.cv_results_["params"]):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std, params))
    results[json.dumps(params)] = (mean, std)

with open(f"results/precisions_{args.train_size}_{args.n_folds}" + args.reduced * "_reduced" + ".json", "w") as result_file:
    json.dump(results, result_file, indent=4)

print(f"The best parameters: {grid.best_params_}")
pred_labels = grid.predict(features)

print("VN | FP \n FN | VP")
print(confusion_matrix(labels, pred_labels))

# with open(f"best_svm_{args.train_size}_{args.n_folds}" + args.reduced * "_reduced" + ".pkl", "wb") as pickle_file:
#     pickle.dump(grid.best_estimator_, pickle_file)