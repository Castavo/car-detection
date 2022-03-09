from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import ParameterGrid
from multiprocessing import Pool
import numpy as np
import pickle, os, argparse, json
from tqdm import tqdm


N_PROCESSES = int(os.getenv("SLURM_CPUS_PER_TASK", 1))

parser = argparse.ArgumentParser()
parser.add_argument("--linear", action="store_true")
parser.add_argument("--res_path", help="The name of the pickle file to write the svm and params to")

args = parser.parse_args()


def fit_and_test(svm_params):
    with open("data/hog/features/train_64_8.pkl", "rb") as pickle_file:
        train_features, train_labels = pickle.load(pickle_file)

    with open("data/hog/features/val_64_8.pkl", "rb") as pickle_file:
        val_features, val_labels = pickle.load(pickle_file)
    if svm_params["kernel"] == "linear":
        svm_params.pop("kernel")
        svm = LinearSVC(
            fit_intercept=True, 
            dual=len(train_labels) > len(train_features[0]), 
            verbose=0,
            max_iter=2000,
            **svm_params, 
        )
    else:
        svm = SVC(
            verbose=0,
            max_iter=2000,
            **svm_params, 
        )
    svm.fit(train_features, train_labels)
    pred_train_labels = svm.predict(train_features)
    pred_val_labels = svm.predict(val_features)
    return svm_params, (f1_score(train_labels, pred_train_labels), confusion_matrix(train_labels, pred_train_labels),
    f1_score(val_labels, pred_val_labels), confusion_matrix(val_labels, pred_val_labels))

if args.linear:
    param_grid = {
        "C": np.logspace(-2, 1, 7), 
        "class_weight": [{0: 1, 1: w} for w in [1, .8, .5, .2]],
        "kernel": ["linear"]
    }
else:
    param_grid = {
        "C": np.logspace(-2, 1, 7), 
        "gamma": np.logspace(-2, 1, 7),
        "kernel": ["rbf"]
    }
grid = ParameterGrid(param_grid)


pool = Pool(N_PROCESSES)
results = pool.imap_unordered(
    fit_and_test,
    grid
)

pool.close()

results = list(tqdm(results, total=len(list(grid))))

for svm_param, result in results:
    print(f"--- for params {svm_param} ------")
    print(f"Train f1-score: {result[0]}")
    print(result[1])
    print(f"Val f1-score: {result[2]}")
    print(result[3])
    print("[TN, FP], [FN, TP]")

dest_name = args.res_path or "results/grid_search_64_8.json"
with open(dest_name, "w") as json_file:
    json.dump(results, json_file)

