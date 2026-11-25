import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

def k_means():
    df = pd.read_csv("data/FP_data2.csv")

    features = [
        "Energy Efficient Appliances",
        "Energy Saving Practices",
        "Renewable Energy Usage"
    ]

    X = df[features].copy()

    # Keep an encoder for each feature
    encoders = {}
    for col in X.columns:
        encoders[col] = LabelEncoder()
        X[col] = encoders[col].fit_transform(X[col].astype(str))

    X_scaled = StandardScaler().fit_transform(X)

    k = 3
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    X["Clusters"] = model.fit_predict(X_scaled)

    summary = X.groupby("Clusters")[features].mean()

    binary_col = ["Energy Efficient Appliances", "Renewable Energy Usage"]
    summary[binary_col] = summary[binary_col] * 100

    st.dataframe(summary, use_container_width=True)

    st.subheader("🔮 Cluster Feature Heatmap")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.heatmap(summary, annot=True, cmap="YlGnBu", ax=ax)
    ax.set_title("Cluster Feature Heatmap")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    return df
