import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# Load Crop Data from CSV
def load_crop_data_from_csv(csv_file):
    try:
        # Read the dataset
        df = pd.read_csv(csv_file)

        # Check if all required columns exist
        required_columns = ["N", "P", "K", "temperature", "ph", "rainfall", "humidity", "label"]
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"The CSV file must contain the following columns: {', '.join(required_columns)}")

        return df
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return None


# Preprocess Data
def preprocess_data(df):
    # Encode the target variable (crop names)
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["label"])

    X = df.drop("label", axis=1).values  # Features: N, P, K, Temperature, pH, Rainfall, Humidity
    y = df["label"].values  # Target: Crop label

    return X, y, label_encoder


# Define DecisionTree Class
class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        if len(set(y)) == 1:  # All labels are the same
            return {'label': y[0]}

        if self.max_depth and depth >= self.max_depth:
            return {'label': self._most_common_label(y)}

        if len(X) < self.min_samples_split:
            return {'label': self._most_common_label(y)}

        best_split = self._best_split(X, y)
        if not best_split:
            return {'label': self._most_common_label(y)}

        left_tree = self._build_tree(X[best_split['left_indices']], y[best_split['left_indices']], depth + 1)
        right_tree = self._build_tree(X[best_split['right_indices']], y[best_split['right_indices']], depth + 1)

        return {'feature': best_split['feature'], 'threshold': best_split['threshold'],
                'left': left_tree, 'right': right_tree}

    def _best_split(self, X, y):
        best_gini = float('inf')
        best_split = {}
        n_features = X.shape[1]

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = ~left_indices

                if np.sum(left_indices) >= self.min_samples_leaf and np.sum(right_indices) >= self.min_samples_leaf:
                    left_gini = self._gini_impurity(y[left_indices])
                    right_gini = self._gini_impurity(y[right_indices])
                    gini = (len(y[left_indices]) * left_gini + len(y[right_indices]) * right_gini) / len(y)

                    if gini < best_gini:
                        best_gini = gini
                        best_split = {'feature': feature, 'threshold': threshold,
                                      'left_indices': left_indices, 'right_indices': right_indices}
        return best_split

    def _gini_impurity(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def _most_common_label(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def predict(self, X):
        return [self._predict_one(x, self.tree) for x in X]

    def _predict_one(self, x, tree):
        if 'label' in tree:
            return tree['label']

        feature_value = x[tree['feature']]
        if feature_value <= tree['threshold']:
            return self._predict_one(x, tree['left'])
        else:
            return self._predict_one(x, tree['right'])


# Define RandomForest Class
class RandomForest:
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.max_features = self.max_features or n_features  # Use all features if not specified

        for _ in range(self.n_estimators):
            # Bootstrap sampling: sample with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[indices], y[indices]

            # Select a random subset of features
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)

            # Train a decision tree on the subset
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                min_samples_leaf=self.min_samples_leaf)
            tree.fit(X_sample[:, feature_indices], y_sample)
            self.trees.append((tree, feature_indices))

    def predict(self, X):
        tree_predictions = []
        for tree, feature_indices in self.trees:
            predictions = tree.predict(X[:, feature_indices])
            tree_predictions.append(predictions)

        # Use majority voting to determine the final prediction
        tree_predictions = np.array(tree_predictions)
        majority_votes = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0, arr=tree_predictions
        )
        return majority_votes


# Main Crop Prediction Program
if __name__ == "__main__":
    csv_file = "Crop_recommendation.xlsx"  # CSV file containing the crop dataset
    crop_data = pd.read_excel(csv_file)

    if crop_data is not None:
        # Preprocess dataset
        X, y, crop_label_encoder = preprocess_data(crop_data)

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Train Random Forest Model
        rf = RandomForest(n_estimators=10, max_depth=5, min_samples_split=2, min_samples_leaf=1)
        rf.fit(X_train, y_train)

        # Make predictions
        y_pred = rf.predict(X_test)

        # Evaluate model accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Crop Prediction Model Accuracy: {accuracy:.4f}")

        # Example Prediction
        test_input = np.array([[80, 35, 50, 28, 6.5, 250, 72]])  # Example input
        predicted_crop = rf.predict(test_input)
        predicted_crop_name = crop_label_encoder.inverse_transform(predicted_crop)
        print(f"Predicted Crop for {test_input.tolist()}: {predicted_crop_name[0]}")
