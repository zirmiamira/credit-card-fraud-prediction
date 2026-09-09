import sys
from pathlib import Path
import pandas as pd

# Allow importing predict.py from the src folder
sys.path.append(str(Path(__file__).resolve().parent))

from predict import predict_fraud


# Load one transaction from the processed dataset
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "creditcard_clean.csv"

df = pd.read_csv(DATA_PATH)

transaction = df.drop(columns="Class").iloc[0].to_dict()

result = predict_fraud(transaction)

print(result)