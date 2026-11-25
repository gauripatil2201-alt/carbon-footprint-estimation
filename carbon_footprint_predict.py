import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import streamlit as st

def carbon_footprint():
    df = pd.read_csv("c:/Users/Chirag Patil/OneDrive/Desktop/gauri/field project/FP_data2.csv")
    
    
    ele_map = {
        "Grid Electricity  ": 2.2,
        "Both Grid and Solar Electricity": 0.8,
        "Solar Electricity   ": 0.1,
        "Other": 1.5
    }

    tran_map = {
        "Private car": 3.0,
        "Two- wheeler": 0.8,
        "Public Transportation": 0.2,
        "Walking & Cycling": 0.0
    }

    pri_vec_map = {
        "Always": 1.2, "Sometimes": 0.6, "Rarely": 0.2, "Never": 0.0
    }

    food_map = {
        "Always": 1.8,"Sometimes": 1.0,
        "Rarely": 0.5, "Never": 0.2
    }

    waste_map = {"Yes": -0.3, "No": 0.0}

    energy_cons_map = {
        "1- very low": 0.8,
        "2- low": 1.5,
        "3- moderate": 2.8,
        "4- high": 4.5,
        "5- very high": 6.2
    }

    df["saves_lights"] = (
        df["Energy Saving Practices"]
        .str.contains("Switching off lights", na=False)
        .astype(int) * -0.2
    )
    df["limits_ac"] = (
        df["Energy Saving Practices"]
        .str.contains("Limiting A.C", na=False)
        .astype(int) * -0.4
    )
    df["uses_led"] = (
        df["Energy Saving Practices"]
        .str.contains("LED lighting", na=False)
        .astype(int) * -0.2
    )
    df["total_savings_credit"] = (
        df["saves_lights"] + df["limits_ac"] + df["uses_led"]
    )

    df["energy_footprint"] = (
        df["Electricity Source"].map(ele_map).fillna(1.5)
        + df["Energy Consumption Rating"].map(energy_cons_map).fillna(2.8)
    )

    df["transport_footprint"] = (
        df["Private Vehicle Usage"].map(pri_vec_map).fillna(0.5)
        * df["Transportation Mode"].map(tran_map).fillna(0.5)
    )

    df["lifestyle_footprint"] = (
        df["Packaged Food Consumption"].map(food_map).fillna(0.8)
        + df["Waste Separation"].map(waste_map).fillna(0.0)
    )

    df["Carbon_Footprint"] = (
        df["energy_footprint"]
        + df["lifestyle_footprint"]
        + df["transport_footprint"]
    )

    # Summary
    output = df.groupby("Household Type")["Carbon_Footprint"].agg(
        Total_Emissions="sum",
        Mean_Average="mean",
        Median_Value="median"
    ).reset_index()

    output.columns = [
        "Household Type",
        "Carbon Footprint(tco2e)",
        "Mean(avg footprint)",
        "Median(typical footprint)"
    ]

    mean = df["Carbon_Footprint"].mean()
    median = df["Carbon_Footprint"].median()
    sample_size = len(df)

    summary = pd.DataFrame({
        "Metric": ["Survey Sample Size", "Dataset Mean", "Dataset Median"],
        "Value": [
            f"{sample_size} Households",
            f"{mean} tco2e/year",
            f"{median} tco2e/year"
        ]
    })

    

    # Separate encoder for every input feature
    feature_cols = [
        "Electricity Source",
        "Transportation Mode",
        "Private Vehicle Usage",
        "Packaged Food Consumption",
        "Waste Separation",
        "Energy Consumption Rating"
    ]

    df_encoded = df.copy()
    encoders = {}

    for col in feature_cols:
        encoders[col] = LabelEncoder()
        df_encoded[col] = encoders[col].fit_transform(
            df_encoded[col].astype(str)
        )

    X = df_encoded[feature_cols]
    y = df_encoded["Carbon_Footprint"].apply(
        lambda x: 1 if x > df["Carbon_Footprint"].median() else 0
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    return model, encoders, X_test, y_test ,output,summary
carbon_footprint()