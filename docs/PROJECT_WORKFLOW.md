# Project Workflow

1. Select and study the base research paper.
2. Reproduce the paper's Random Forest stock-return prediction workflow.
3. Use technical indicators as input features.
4. Predict 20-day excess return relative to a benchmark.
5. Evaluate using MAE, MSE, RMSE, MAPE, R2 and directional accuracy.
6. Compare the reproduced result with the metrics reported in the paper.
7. Train an Extra Trees model using time-aware validation.
8. Compare both models and document limitations.

Run demo: python3 run_project.py --demo
Run live experiment: python3 run_project.py --live
Dashboard: streamlit run app.py
