import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--outputfile", "-o", type=str, help="output file name")
args = parser.parse_args()

DATA_DIR = "/Users/luckysrivastava/Workspace/data"
DATASET_NAME = "DocumentImages"
DATASET_LOCATION = os.path.join(DATA_DIR, DATASET_NAME)
LABELS = os.listdir(DATASET_LOCATION)

TRAIN_TEST_SPLIT = 0.1

outputfile = args.outputfile
train_out = outputfile.replace(".txt", "_train.txt")
test_out = outputfile.replace(".txt", "_test.txt")

with open(train_out, "w") as f:
    f.write("filepath, label\n")

with open(test_out, "w") as f:
    f.write("filepath, label\n")

for (root_path, dirs, _) in os.walk(DATASET_LOCATION):
    for dirname in dirs:
        subdir = os.path.join(root_path, dirname)
        split_idx = len(os.listdir(subdir)) * TRAIN_TEST_SPLIT
        for idx, filename in enumerate(os.listdir(subdir)):
            fullfilepath = os.path.join(subdir, filename)
            label = dirname
            if idx <= split_idx:
                outputfile = test_out
            else:
                outputfile = train_out
            with open(outputfile, "a") as f:
                f.write(fullfilepath + "," + label + "\n")
