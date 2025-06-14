import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from scipy.stats import pearsonr

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
st.markdown("Summary Table of Continuous Full Sample")
st.dataframe(styled_table)

#Correlatation

def corr_with_pvalues(df):
    cols = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']
    corr_matrix = pd.DataFrame(index=cols, columns=cols)
    pval_matrix = pd.DataFrame(index=cols, columns=cols)

    for col1 in cols:
        for col2 in cols:
            r, p = pearsonr(df[col1], df[col2])
            corr_matrix.loc[col1, col2] = round(r, 3)
            pval_matrix.loc[col1, col2] = round(p, 5)

    return corr_matrix.astype(float), pval_matrix.astype(float)

# Assume df is already loaded in your Streamlit app

corrs, pvals = corr_with_pvalues(df)
mask_significant = pvals < 0.05
masked_corrs = corrs.where(mask_significant)

plt.figure(figsize=(8, 6))
sns.heatmap(
    masked_corrs,
    annot=True,
    cmap='coolwarm',
    center=0,
    vmin=-1, vmax=1,
    linewidths=0.5,
    linecolor='gray',
    mask=~mask_significant,
    cbar_kws={'label': 'Pearson Correlation'}
)
plt.title('Significant Pearson Correlations (p < 0.05)')
plt.tight_layout()

# Instead of plt.show(), use:
st.pyplot(plt.gcf())

