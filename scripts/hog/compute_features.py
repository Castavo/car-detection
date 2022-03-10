from scripts.utils import extract_frames_info, SIMILAR_PREFIXES, NIGHT_PREFIXES, remove_small_bb
from .classifier import HOGClassifier
import os, pickle


N_PROCESSES = int(os.getenv("SLURM_CPUS_PER_TASK", 1))

frames_info = extract_frames_info("data/train.csv", skip_prefixes=[SIMILAR_PREFIXES[0]] + NIGHT_PREFIXES)
frames_info = remove_small_bb(frames_info)

hog_params = dict(pixels_per_cell=(8, 8), orientations=8, cells_per_block=(3, 3))

classifier = HOGClassifier(hog_params, None, goal_shape=(64, 64))

features, labels = classifier.features_labels(frames_info, N_PROCESSES, augment=True, n_negatives=6)

os.makedirs("data/hog/features", exist_ok=True)

with open("data/hog/features/train_64_8_no_night.pkl", "wb") as pickle_file:
    pickle.dump((features, labels), pickle_file)

print(len(features))

val_frames_info = extract_frames_info("data/train.csv", only_prefixes=[SIMILAR_PREFIXES[0]])  
val_frames_info = remove_small_bb(val_frames_info)
val_features, val_labels = classifier.features_labels(val_frames_info, N_PROCESSES, augment=False, n_negatives=8)

with open("data/hog/features/val_64_8.pkl", "wb") as pickle_file:
    pickle.dump((val_features, val_labels), pickle_file)

print(len(val_features))