import os

from preprocess import preprocess
from build_model import build
from evaluate_model import evaluate


def create_subdirectory_if_not_exists(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
        print("Directory", dir_name, "created.")
    else:
        print("Directory", dir_name, "already exists.")


# Create output directory to store preprocessed data and trained model
create_subdirectory_if_not_exists("out")

# Define file locations
train_data = "out/preprocessed_train.npz"
test_data = "out/preprocessed_test.npz"
model_file = "out/model.h5"

if not os.path.isfile(train_data) or not os.path.isfile(test_data):
    # Preprocess data
    preprocess(train_data=train_data, test_data=test_data)
    print("Data preprocessed and saved locally.")
else:
    print("Preprocessed data exists.")

# Train model
build(train_data=train_data, save_file=model_file)

# Evaluate model
scores = evaluate(model_file, test_data)
print("Final scores:", scores)
