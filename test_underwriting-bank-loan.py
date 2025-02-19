import pandas as pd
from model.underwriting_model import LoanUnderwritingModel

# Load dataset
df = pd.read_csv("data/bank-loan.csv")

# Handle missing values
df.dropna(inplace=True)

# Split features and target
X = df.drop(columns=["default"])
y = df["default"]

# Initialize and train model
model = LoanUnderwritingModel()
results = model.train(X, y)
