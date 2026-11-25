import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def train_model():
    # 1. Load the cleaned dataset
    df = pd.read_csv("data/FP_data2.csv")

    # 2. Select features
    X = df[[
        "Household Type",
        "Residence Type",
        "Age Group",
        "Energy Efficient Appliances"
    ]].copy()

    # 3. Target variable
    y = df["Renewable Energy Usage"].astype(str)

    # 4. Use a separate encoder for every feature
    encoders = {}
    for col in X.columns:
        encoders[col] = LabelEncoder()
        X[col] = encoders[col].fit_transform(X[col].astype(str))

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)
    encoders["_target"] = target_encoder

    # 5. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 6. Build Decision Tree
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # No plot here.
    # The tree is created only on the Model insights page if needed.

    return model, encoders, X_test, y_test

def show_decision_tree(model, feature_names):
    fig, ax = plt.subplots(figsize=(5, 6))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=["No", "Yes"],
        filled=True,
        ax=ax
    )
    fig.tight_layout()
    return fig
