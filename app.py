import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo

# fetch dataset
heart_failure_clinical_records = fetch_ucirepo(id=519)

# data (as pandas dataframes)
X = heart_failure_clinical_records.data.features
y = heart_failure_clinical_records.data.targets
df = pd.concat([X, y], axis=1)

st.title("Heart Failure Survival Prediction Dashboard")

# Show dataset
if st.checkbox("Show raw data"):
    st.dataframe(df)