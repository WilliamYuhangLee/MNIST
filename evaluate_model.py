import numpy as np

from keras.models import load_model


def evaluate(model_file="out/model.h5", test_data="out/preprocessed_test.npz"):
    # Load model and test data
    model = load_model(filepath=model_file)
    test_data = np.load(test_data)
    x_test, y_test = test_data["x_test"], test_data["y_test"]

    # Evaluate model
    scores = model.evaluate(x=x_test, y=y_test)
    return scores

if __name__ == "__main__":
    print(evaluate())
