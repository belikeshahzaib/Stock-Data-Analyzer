1. Install Python (3.10+) if not already installed.

2. Install required packages:
   pip install streamlit pandas yfinance matplotlib seaborn plotly numpy scikit-learn 

3. Make sure the following files are in the same folder:
   - app.py
   - clean_data.py
   - feature_engineering.py
   - eda.py
   - model_training.py

4. Run the app:
   python -m streamlit run app.py

5. The sidebar allows you to select:
   - Train Stock
   - Test Stock
   - Start and End Dates
   - Prediction Horizon (days)

6. The app automatically downloads data and saves:
   - Raw CSVs: data/train_raw.csv, data/test_raw.csv
   - Cleaned CSVs: data/train_clean.csv, data/test_clean.csv

7. View EDA, predictions, and metrics in the app.