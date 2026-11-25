import streamlit as st
import pandas as pd
from modules.decisiontree import train_model,show_decision_tree
from modules.kmeans import k_means
from modules.field_project import demographics, energy_usage, waste, transportation, awareness 
from modules.carbon_footprint_predict import carbon_footprint
from modules.suggestion_analysis import suggestion_plot
import matplotlib.pyplot as plt 

# Page config
st.set_page_config(page_title="Carbon Footprint Estimation", page_icon="🎓", layout="wide")

st.sidebar.title("📊 Data Analysis")
Statistics = st.sidebar.selectbox(
    "Statistics",
    ["Home", "Demographics", "Energy and Appliances", "Waste and Sustainability",
     "Transportation and Consumption", "Awareness","Suggestions", "Model insights",
     "Clustering", "Carbon Footprint"]
)

df = pd.read_csv("data/FP_data2.csv")

if Statistics == "Home":
    st.header("🏠 Project Dashboard:Field-Based Carbon Footprint Estimation for Households")
    st.subheader("Project Overview")
    st.write("This project focuses on studying household activities and lifestyle practices that contribute to carbon-footprint generation./n" \
        " Primary data is collected through a Google Form survey from households, covering areas such as electricity consumption, cooking fuel, transportation, energy-efficient appliances, renewable-energy usage, waste management, energy-saving practices, and awareness of carbon-footprint reduction.")
    st.write("The collected data is analysed using Python and Excel through data cleaning, statistical analysis, percentages, charts, and visualization. Text mining is also performed on open-ended responses to identify commonly mentioned words and suggestions related to reducing household carbon footprints. /n" \
        "The project aims to identify patterns in household behaviour and understand areas where more sustainable practices can be encouraged.")

    st.subheader("PROJECT OBJECTIVE")
    st.write("The main objective of this project is to study household practices and behaviours that are related to carbon-footprint generation./n" \
        " The survey collects information about electricity usage, cooking fuel,transportation, waste management, energy-saving practices, renewable energy adoption, food habits, and awareness of carbon-footprint reduction./n" \
        "The collected responses are analysed using data analysis and basic data-mining techniques in Python and Excel to identify patterns, trends, and commonly reported practices among households")
    st.subheader("Analysis Performed:")
    st.write("A. Descriptive Analysis(pie chart, bar chart)")
    st.write("B. Comparative Analysis")
    st.write("C. Text Mining")
    st.write("D. Clustering(KMeans)")
elif Statistics == "Demographics":
    demographics()
    st.write("👉 Most respondents live in **nuclear families** and in **apartments**.")

elif Statistics == "Energy and Appliances":
    energy_usage()
    st.write("👉 Energy-efficient appliances are more common than renewable energy adoption.")

elif Statistics == "Waste and Sustainability":
    waste()
    st.write("👉 Joint families generate more mixed waste compared to nuclear families.")

elif Statistics == "Transportation and Consumption":
    transportation()
    st.write("👉 Majority rely on public transport, but private vehicle usage is significant.")

elif Statistics == "Awareness":
    awareness()
    st.write("👉 Awareness levels vary widely, but most rate their energy consumption as moderate.")
    st.write("👉 Main contributers to carbon footprint are waste generation and transportation .")

elif Statistics == "Model insights":
    model, encoders, X_test, y_test = train_model()

    st.header("🔮 Decision Tree Prediction")
    st.metric("Decision Tree Accuracy", f"{model.score(X_test, y_test):.2f}")
    feature_names = X_test.columns.tolist()
    fig = show_decision_tree(model,feature_names=feature_names)   
    st.pyplot(fig)                  

    household = st.selectbox("Household Type", df["Household Type"].unique())
    residence = st.selectbox("Residence Type", df["Residence Type"].unique())
    age_group = st.selectbox("Age Group", df["Age Group"].unique())
    appliances = st.selectbox("Energy Efficient Appliances", df["Energy Efficient Appliances"].unique())

    if st.button("Predict"):
        input_data = pd.DataFrame({
            "Household Type": [household],
            "Residence Type": [residence],
            "Age Group": [age_group],
            "Energy Efficient Appliances": [appliances]
        })

        for col in input_data.columns:
            input_data[col] = encoders[col].transform(input_data[col].astype(str))

        prediction = model.predict(input_data)[0]
        result = "Yes (Renewable Energy Used)" if prediction == 1 else "No (Not Using Renewable Energy)"
        st.success(f"Prediction: {result}")

elif Statistics == "Clustering":
    k_means()
    st.subheader("🔮 Cluster Insights")
    st.write("👉 Interpretation:")
    st.write("- Cluster 0:Strong energy-saving practices and higher renewable usage.")
    st.write("- Cluster 1: Higher appliance efficiency but moderate renewable usage. ")
    st.write("- Cluster 2: Low efficiency and low renewable adoption.")

elif Statistics == "Carbon Footprint":
    cf_model, cf_encoders, cf_X_test, cf_y_test ,cf_output,cf_summary= carbon_footprint()

    st.subheader("🔮 Carbon Footprint Prediction")
    st.write("📋 CARBON FOOTPRINT SUMMARY BY HOUSEHOLD PROFILE")
    st.dataframe(cf_output, use_container_width=True)
    st.write("📊 DATASET OVERALL STATISTICAL BASELINE")
    st.dataframe(cf_summary, use_container_width=True)
    st.write("Decision Tree Accuracy:", cf_model.score(cf_X_test, cf_y_test))

    with st.form("prediction_form"):
        electricity_source = st.selectbox(
            "Electricity Source",
            ["Grid Electricity  ", "Both Grid and Solar Electricity", "Solar Electricity   ", "Other"]
        )
        transport_mode = st.selectbox(
            "Transportation Mode",
            ["Private car", "Two- wheeler", "Public Transportation", "Walking & Cycling"]
        )
        private_vehicle_usage = st.selectbox(
            "Private Vehicle Usage",
            ["Always", "Sometimes", "Rarely", "Never"]
        )
        packaged_food = st.selectbox(
            "Packaged Food Consumption",
            ["Always","Sometimes", "Rarely", "Never"]
        )
        waste_sep = st.selectbox("Waste Separation", ["Yes", "No"])
        energy_rating = st.selectbox(
            "Energy Consumption Rating",
            ["1- very low", "2- low", "3- moderate", "4- high", "5- very high"]
        )

        submit_button = st.form_submit_button("Predict")

    if submit_button:
        input_data = pd.DataFrame({
            "Electricity Source": [electricity_source],
            "Transportation Mode": [transport_mode],
            "Private Vehicle Usage": [private_vehicle_usage],
            "Packaged Food Consumption": [packaged_food],
            "Waste Separation": [waste_sep],
            "Energy Consumption Rating": [energy_rating]
        })

        st.write("✅ Input data ready for prediction:")
        st.dataframe(input_data)

        for col in input_data.columns:
            input_data[col] = cf_encoders[col].transform(input_data[col].astype(str))

        prediction = cf_model.predict(input_data)[0]
        result = "High Footprint" if prediction == 1 else "Low Footprint"
        st.success(f"Prediction: {result}")
elif Statistics=="Suggestions":
    filtered_words, common_words, fig, word_count = suggestion_plot()
    st.subheader("Sample Filtered Words")
    st.write(filtered_words[:20])
    st.subheader("Top 10 Most Frequent Words")
    st.write(common_words)
    st.pyplot(fig)
    st.subheader("Specific Word Counts")
    st.write("Solar:", word_count["solar"])
    st.write("Electricity:", word_count["electricity"])