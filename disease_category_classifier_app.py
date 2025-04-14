# Cell Summary: Write the Streamlit application code to a Python file.
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

st.title("Disease Category Classification Model Explorer")
st.write("""
Explore the performance of K-Nearest Neighbors (KNN) and Logistic Regression models
trained on different feature representations (TF-IDF vs. One-Hot) to predict **disease categories**.
Results are based on 5-fold cross-validation performed **only on categories with sufficient samples (>= 5)**.
Select parameters below to view the results.
""")

@st.cache_data
def load_results():
    RESULTS_FILE = "model_evaluation_results_categories.csv"
    try:
        df = pd.read_csv(RESULTS_FILE)
        df['k'] = pd.to_numeric(df['k'], errors='coerce').fillna(0).astype(int)
        return df
    except FileNotFoundError:
        st.error(f"Error: `{RESULTS_FILE}` not found. Please run the Jupyter notebook first.")
        return None
    except Exception as e:
        st.error(f"An error occurred loading results: {e}")
        return None

results_df = load_results()

if results_df is not None:
    st.sidebar.header("Model Selection")

    model_type = st.sidebar.selectbox("Select Model Type:", sorted(results_df['Model'].unique()))
    encoding_type = st.sidebar.selectbox("Select Feature Encoding:", sorted(results_df['Encoding'].unique()))

    k_value = 0
    metric_type = 'N/A'
    if model_type == 'KNN':
        k_values_available = sorted([k for k in results_df['k'].unique() if k > 0])
        if k_values_available:
             k_value = st.sidebar.select_slider("Select k (Number of Neighbors):", options=k_values_available, value=k_values_available[len(k_values_available)//2])
             metrics_available = sorted(results_df[(results_df['Model'] == 'KNN') & (results_df['k'] == k_value)]['Metric'].dropna().unique())
             if metrics_available:
                 metric_type = st.sidebar.selectbox("Select Distance Metric:", options=metrics_available)
             else:
                  st.sidebar.warning(f"No metrics found for KNN with k={k_value}.")
                  metric_type = None
        else:
             st.sidebar.warning("No valid k values found for KNN in the results.")
             metric_type = None

    filtered_results = results_df[(results_df['Model'] == model_type) & (results_df['Encoding'] == encoding_type)]

    if model_type == 'KNN':
        if k_value > 0 and metric_type is not None:
             filtered_results = filtered_results[(filtered_results['k'] == k_value) & (filtered_results['Metric'] == metric_type)]
        else:
             filtered_results = pd.DataFrame()

    st.header("Selected Model Performance (5-Fold CV Averages on Filtered Categories)")
    if not filtered_results.empty:
        st.subheader(f"Configuration: {model_type} on {encoding_type}" + (f" (k={k_value}, Metric={metric_type})" if model_type == 'KNN' and k_value > 0 else ""))
        col1, col2, col3 = st.columns(3)
        result_row = filtered_results.iloc[0]
        col1.metric("Average Accuracy", f"{result_row['Avg Accuracy']:.4f}", f"±{result_row['Std Accuracy']:.4f}")
        col2.metric("Average F1-score (Weighted)", f"{result_row['Avg F1-score']:.4f}", f"±{result_row['Std F1-score']:.4f}")
        col3.metric("Average Precision (Weighted)", f"{result_row['Avg Precision']:.4f}")
        st.dataframe(filtered_results[['Encoding', 'Model', 'k', 'Metric', 'Avg Accuracy', 'Avg Precision', 'Avg Recall', 'Avg F1-score', 'Std F1-score', 'Std Accuracy']].style.format({
             'Avg Accuracy': '{:.4f}', 'Std Accuracy': '{:.4f}', 'Avg Precision': '{:.4f}', 'Avg Recall': '{:.4f}',
             'Avg F1-score': '{:.4f}', 'Std F1-score': '{:.4f}'
        }))
    else:
        st.warning("No results found for the selected configuration.")

    st.header("Full Results Table (Category Prediction)")
    st.dataframe(results_df.style.format({
             'Avg Accuracy': '{:.4f}', 'Std Accuracy': '{:.4f}', 'Avg Precision': '{:.4f}', 'Avg Recall': '{:.4f}',
             'Avg F1-score': '{:.4f}', 'Std F1-score': '{:.4f}'
        }))
    st.caption("Note: Performance metrics reflect the model's ability to classify diseases into broader categories, evaluated only on categories with 5 or more samples.")
else:
    st.error("Could not load model evaluation results. Cannot display the application.")

print("Streamlit app code written to disease_category_classifier_app.py")