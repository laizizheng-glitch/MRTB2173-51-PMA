# MRTB2173-51-PMA

## Project Title

Telecommunication Customer Churn Agile Data Science Project

## Dataset

Dataset source:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Expected file:

`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Sprint Structure

- Sprint 1: Dataset preparation, EDA and data quality checks
- Sprint 2: Feature engineering and baseline Logistic Regression model
- Sprint 3: Improved Random Forest model and model comparison
- Sprint 4: Streamlit dashboard, monitoring, validation, testing and CI/CD
- Final: Final submission version-control commit

## Run Project

```bash
pip install -r requirements.txt
python src/sprint1_data_preparation.py
python src/sprint2_baseline_model.py
python src/sprint3_improved_model.py
python src/sprint4_dashboard_monitoring.py
python src/validation_script.py
pytest tests/ -v
streamlit run dashboard/app.py
```
