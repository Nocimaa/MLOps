import csv
import pickle
from pathlib import Path

CSV_PATH = Path(__file__).with_name("houses.csv")
MODEL_PATH = Path(__file__).with_name("linear_model.joblib")
FEATURE_NAMES = ["size", "nb_rooms", "garden"]


def load_dataset(path):
    features = []
    targets = []
    with open(path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                x = [float(row[name]) for name in FEATURE_NAMES]
            except ValueError:
                continue
            y = float(row["price"])
            features.append(x)
            targets.append(y)
    return features, targets


def solve_linear_system(matrix, vector):
    n = len(vector)
    # build augmented matrix
    aug = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for i in range(n):
        pivot = aug[i][i]
        if abs(pivot) < 1e-12:
            for j in range(i + 1, n):
                if abs(aug[j][i]) > abs(pivot):
                    aug[i], aug[j] = aug[j], aug[i]
                    pivot = aug[i][i]
                    break
        if abs(pivot) < 1e-12:
            raise ValueError("matrix is singular")
        for j in range(i, n + 1):
            aug[i][j] /= pivot
        for k in range(n):
            if k == i:
                continue
            factor = aug[k][i]
            for j in range(i, n + 1):
                aug[k][j] -= factor * aug[i][j]
    return [aug[i][-1] for i in range(n)]


def train_least_squares(features, targets):
    n_samples = len(features)
    if n_samples == 0:
        raise ValueError("No samples to train")
    n_features = len(features[0])
    xtx = [[0.0] * (n_features + 1) for _ in range(n_features + 1)]
    xty = [0.0] * (n_features + 1)
    for x, y in zip(features, targets):
        augmented = [1.0] + x
        for i in range(n_features + 1):
            xty[i] += augmented[i] * y
            for j in range(n_features + 1):
                xtx[i][j] += augmented[i] * augmented[j]
    theta = solve_linear_system(xtx, xty)
    intercept = theta[0]
    coefficients = theta[1:]
    return {"coef_": coefficients, "intercept_": intercept}


def main():
    features, targets = load_dataset(CSV_PATH)
    model = train_least_squares(features, targets)
    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)
    print(f"Trained linear model saved to {MODEL_PATH}")
    print("Coefficients:", model["coef_"])
    print("Intercept:", model["intercept_"])


if __name__ == "__main__":
    main()
