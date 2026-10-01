# Stock Market Analysis using Machine Learning

**Course:** SEML3211 — Foundation of Machine Learning  
**Project:** Stock Market Analysis

## Project objective

This project follows a research-paper reproduction workflow:

1. Study a published stock-return prediction paper.
2. Reproduce its Random Forest-based approach using technical indicators.
3. Evaluate the reproduced model with regression metrics.
4. Compare the observed results with the values reported in the paper.
5. Propose Extra Trees as an alternative ensemble model.
6. Use time-aware validation and compare the models.

## Base paper

Mohapatra, S., Mukherjee, R., Roy, A., Sengupta, A., & Puniyani, A. (2022).  
**Can Ensemble Machine Learning Methods Predict Stock Returns for Indian Banks Using Technical Indicators?**  
Journal of Risk and Financial Management, 15(8), 350.

DOI: https://doi.org/10.3390/jrfm15080350

See `BASE_PAPER_LINK.txt` for the reference and instructions for adding the publisher PDF to the college submission ZIP.

## Models

- Random Forest Regressor — paper reproduction
- Extra Trees Regressor — proposed comparison model

## Features

The implementation uses technical/price-volume features including:

`10by20, 10by40, MAT20, MAT40, MSV10, MSV20, MSV40, POP9, POP19, POP39`

## Evaluation

- MAE
- MSE
- RMSE
- MAPE
- R²
- Directional Accuracy

## Run

Create and activate a virtual environment, then:

```bash
pip install -r requirements.txt
python3 run_project.py --demo
python3 run_project.py --live
```

Optional dashboard:

```bash
streamlit run app.py
```

## Recorded live experiment

See `docs/RESULTS.md` for the observed project-run metrics.

## Submission contents

The college submission should contain:

- Base paper PDF
- Implementation code
- Results/graphs
- Drafted research paper

The repository is intended to keep the reproducible implementation and project documentation in one place.
