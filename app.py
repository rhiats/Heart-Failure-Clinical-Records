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

continuous_features_df = X[['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time']]

summary_continuous_full_sample_df = pd.DataFrame({
    'Median': continuous_features_df.median(),
    'Mean': round(continuous_features_df.mean(), 2),
    'Standard Deviation': round(continuous_features_df.std(), 2)
})

# Highlight where Std > Mean
def highlight_std_gt_mean(row):
    color = 'background-color: lightcoral'
    default = ''
    return [color if row['Standard Deviation'] > row['Mean'] and col in ['Standard Deviation', 'Mean'] else default for col in row.index]

# Apply the style
styled_table = summary_continuous_full_sample_df.style.apply(highlight_std_gt_mean, axis=1)

# Display
st.dataframe(styled_table)