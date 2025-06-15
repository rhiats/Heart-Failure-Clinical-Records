import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import shap

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
st.subheader("Summary Table of Continuous Full Sample", divider="blue")
st.markdown(
    f"<div style='display: flex; justify-content: center'>{styled_table.to_html()}</div>",
    unsafe_allow_html=True
)

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


def evaluation_metrics(y_test, y_pred):
  """
    Evaluate the performance of a model.

    @param y_test: array of true labels
    @param y_pred: array of predicted labels
    @return: Evaluation metrics
  """

  mcc = matthews_corrcoef(y_test, y_pred)

  f1 = f1_score(y_test, y_pred)
  accuracy = accuracy_score(y_test, y_pred)
  recall_s = recall_score(y_test, y_pred)

  tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
  tn_rate = tn/(tn+fp)

  precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
  pr_auc = auc(recall, precision)

  roc_auc = roc_auc_score(y_test, y_pred)

  return mcc, f1, accuracy, recall_s, tn_rate, pr_auc, roc_auc

def avg_evaluation_metrics(eval_arr):
  """
    Average of the evaluation metrics.

    @param eval_arr: array of evaluation metrics
    @return: Average of the evaluation metrics
  """
  return sum(eval_arr) / len(eval_arr)

#One Rule Implementation
class SimpleOneR:
    def fit(self, X, y):
        self.rules = {}
        best_feature = None
        best_accuracy = 0
        self.default_class = y['death_event'].mode()[0]

        for feature in X.columns:
            rules = {}
            for val in X[feature].unique():
                most_common_class = y[X[feature] == val]['death_event'].mode()[0]
                rules[val] = most_common_class

            preds = X[feature].map(rules)
            acc = accuracy_score(y, preds)

            if acc > best_accuracy:
                best_accuracy = acc
                best_feature = feature
                self.rules = rules

        self.best_feature = best_feature

    def predict(self, X):
        return X[self.best_feature].map(self.rules).fillna(self.default_class)

rf_mcc_arr = []
rf_f1_score_arr = []
rf_accuracy_arr = []
rf_tp_rate_arr = []
rf_tn_rate_arr = []
rf_pr_auc_arr = []
rf_pr_auc_arr = []
rf_roc_auc_arr = []


or_mcc_arr = []
or_f1_score_arr = []
or_accuracy_arr = []
or_tp_rate_arr = []
or_tn_rate_arr = []
or_pr_auc_arr = []
or_pr_auc_arr = []
or_roc_auc_arr = []

nb_mcc_arr = []
nb_f1_score_arr = []
nb_accuracy_arr = []
nb_tp_rate_arr = []
nb_tn_rate_arr = []
nb_pr_auc_arr = []
nb_pr_auc_arr = []
nb_roc_auc_arr = []

dt_mcc_arr = []
dt_f1_score_arr = []
dt_accuracy_arr = []
dt_tp_rate_arr = []
dt_tn_rate_arr = []
dt_pr_auc_arr = []
dt_pr_auc_arr = []
dt_roc_auc_arr = []

gb_mcc_arr = []
gb_f1_score_arr = []
gb_accuracy_arr = []
gb_tp_rate_arr = []
gb_tn_rate_arr = []
gb_pr_auc_arr = []
gb_pr_auc_arr = []
gb_roc_auc_arr = []

ann_mcc_arr = []
ann_f1_score_arr = []
ann_accuracy_arr = []
ann_tp_rate_arr = []
ann_tn_rate_arr = []
ann_pr_auc_arr = []
ann_pr_auc_arr = []
ann_roc_auc_arr = []

mcc_dict = {"Model Type":[],"MCC Score":[]}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train = pd.DataFrame(X_train_scaled, columns = X_train.columns, index = X_train.index)
X_test = pd.DataFrame(X_test_scaled, columns = X_test.columns, index = X_test.index)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mcc, f1, acc, recall, tn_rate, pr_auc, roc_auc = evaluation_metrics(y_test, y_pred)

rf_mcc_arr.append(mcc)
rf_f1_score_arr.append(f1)
rf_accuracy_arr.append(acc)
rf_tp_rate_arr.append(recall)
rf_tn_rate_arr.append(tn_rate)
rf_pr_auc_arr.append(pr_auc)
rf_roc_auc_arr.append(roc_auc)

model = SimpleOneR()
model.fit(X_train, y_train)
y_pred_one_rule = model.predict(X_test)

mcc_or, f1_or, acc_or, recall_or, tn_rate_or, pr_auc_or, roc_auc_or = evaluation_metrics(y_test, y_pred_one_rule)

or_mcc_arr.append(mcc_or)
or_f1_score_arr.append(f1_or)
or_accuracy_arr.append(acc_or)
or_tp_rate_arr.append(recall_or)
or_tn_rate_arr.append(tn_rate_or)
or_pr_auc_arr.append(pr_auc_or)
or_roc_auc_arr.append(roc_auc_or)

gnb = GaussianNB()
y_pred_gb = gnb.fit(X_train, y_train).predict(X_test)

mcc_nb, f1_nb, acc_nb, recall_nb, tn_rate_nb, pr_auc_nb, roc_auc_nb = evaluation_metrics(y_test, y_pred_gb)

nb_mcc_arr.append(mcc_nb)
nb_f1_score_arr.append(f1_nb)
nb_accuracy_arr.append(acc_nb)
nb_tp_rate_arr.append(recall_nb)
nb_tn_rate_arr.append(tn_rate_nb)
nb_pr_auc_arr.append(pr_auc_nb)
nb_roc_auc_arr.append(roc_auc_nb)

dt = tree.DecisionTreeClassifier()
y_pred_dt = dt.fit(X_train, y_train).predict(X_test)

mcc_dt, f1_dt, acc_dt, recall_dt, tn_rate_dt, pr_auc_dt, roc_auc_dt = evaluation_metrics(y_test, y_pred_dt)

dt_mcc_arr.append(mcc_dt)
dt_f1_score_arr.append(f1_dt)
dt_accuracy_arr.append(acc_dt)
dt_tp_rate_arr.append(recall_dt)
dt_tn_rate_arr.append(tn_rate_dt)
dt_pr_auc_arr.append(pr_auc_dt)
dt_roc_auc_arr.append(roc_auc_dt)

mcc_scorer = make_scorer(matthews_corrcoef)

gbc = GradientBoostingClassifier(random_state=42)
param_grid = {
'n_estimators': [50, 100],
'learning_rate': [0.01, 0.1],
'max_depth': [3, 5]
}

grid_search = GridSearchCV(estimator=gbc, param_grid=param_grid, scoring=mcc_scorer, cv=5, n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train)
y_pred_gb = grid_search.predict(X_test)

mcc_gb, f1_gb, acc_gb, recall_gb, tn_rate_gb, pr_auc_gb, roc_auc_gb = evaluation_metrics(y_test, y_pred_gb)

gb_mcc_arr.append(mcc_gb)
gb_f1_score_arr.append(f1_gb)
gb_accuracy_arr.append(acc_gb)
gb_tp_rate_arr.append(recall_gb)
gb_tn_rate_arr.append(tn_rate_gb)
gb_pr_auc_arr.append(pr_auc_gb)
gb_roc_auc_arr.append(roc_auc_gb)

ann = MLPClassifier(max_iter=500, random_state=42)

param_grid = {
'hidden_layer_sizes': [(50,), (100,), (50, 50)],
'activation': ['relu', 'tanh'],
'solver': ['adam', 'sgd'],
'alpha': [0.0001, 0.001],
'learning_rate': ['constant', 'adaptive']
}

grid_search = GridSearchCV(estimator=ann, param_grid=param_grid, scoring=mcc_scorer, cv=5, n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train)

y_pred_ann = grid_search.predict(X_test)

mcc_ann, f1_ann, acc_ann, recall_ann, tn_rate_ann, pr_auc_ann, roc_auc_ann = evaluation_metrics(y_test, y_pred_ann)

ann_mcc_arr.append(mcc_ann)
ann_f1_score_arr.append(f1_ann)
ann_accuracy_arr.append(acc_ann)
ann_tp_rate_arr.append(recall_ann)
ann_tn_rate_arr.append(tn_rate_ann)
ann_pr_auc_arr.append(pr_auc_ann)
ann_roc_auc_arr.append(roc_auc_ann)


mcc_dict["Model Type"].append("Random Forest")
mcc_dict["MCC Score"].append(avg_evaluation_metrics(rf_mcc_arr))

mcc_dict["Model Type"].append("One Rule")
mcc_dict["MCC Score"].append(avg_evaluation_metrics(or_mcc_arr))

mcc_dict["Model Type"].append("Naive Bayes")
mcc_dict["MCC Score"].append(avg_evaluation_metrics(nb_mcc_arr))

mcc_dict["Model Type"].append("Decision Tree")
mcc_dict["MCC Score"].append(avg_evaluation_metrics(dt_mcc_arr))

mcc_dict["Model Type"].append("Gradient Boosting")
mcc_dict["MCC Score"].append(avg_evaluation_metrics(gb_mcc_arr))

mcc_dict["Model Type"].append("Artificial Neural Network")
mcc_dict["MCC Score"].append(avg_evaluation_metrics(ann_mcc_arr))


explainer = shap.Explainer(grid_search.predict, X_test)
shap_values = explainer(X_test)

# Create the beeswarm plot and capture the figure
plt.figure()  # Start a new figure
shap.plots.beeswarm(shap_values, show=False)  # Tell SHAP not to show it immediately

# Then pass the current figure to Streamlit
st.pyplot(plt.gcf())  # gcf = Get Current Figure







