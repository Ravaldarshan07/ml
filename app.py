import streamlit as st
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="Stock Market ML Analysis",page_icon="📈",layout="wide")
st.title("Stock Market Analysis using Machine Learning")
st.caption("SEML3211 — Research-paper reproduction and model comparison")
root=Path(__file__).parent; csv=root/"results/model_results.csv"
if csv.exists():
    df=pd.read_csv(csv); st.subheader("Model Performance"); st.dataframe(df,use_container_width=True)
    c1,c2=st.columns(2)
    if (root/"results/feature_importance.png").exists(): c1.image(str(root/"results/feature_importance.png"),caption="Feature Importance")
    if (root/"results/prediction_scatter.png").exists(): c2.image(str(root/"results/prediction_scatter.png"),caption="Actual vs Predicted")
else: st.info("Run python3 run_project.py --live first to generate results.")
st.markdown("### Project workflow")
st.write("Base paper → Random Forest reproduction → metric comparison → Extra Trees improvement → evaluation.")
