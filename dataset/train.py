import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

# Load data
data_file = 'files_for_training_model/telcom.csv'
df_raw = pd.read_csv(data_file)

# Data transformation function
def Data_transformation_renaming(df_raw):
    """ Rename column names and transform into proper format and return dataframe """
    df_cal = df_raw.copy()

    # Rename columns
    df_cal.rename(columns={'gender':'Gender',
                           'customerID':'CustomerID',
                           'Contract':'ContractType',
                           'InternetService':'InternetServiceType',
                           'tenure':'Tenure'
                          },
                  inplace=True)

    # Map categorical values to numerical values
    df_cal['Partner'] = df_cal.Partner.map({'Yes':1, 'No':0})
    df_cal['Dependents'] = df_cal.Dependents.map({'Yes':1, 'No':0})
    # Add more mapping for other features as needed

    # Data mining
    df_cal['IsContracted'] = df_cal.ContractType.map({'One year':1, 'Two year':1, 'Month-to-month':0})

    # Data transformation for TotalCharges
    df_cal.loc[df_cal['TotalCharges']==' ', 'TotalCharges'] = np.nan
    df_cal['TotalCharges'] = df_cal['TotalCharges'].astype('float64')
    df_cal.loc[df_cal['TotalCharges'].isnull(), 'TotalCharges'] = df_cal['MonthlyCharges'] * df_cal['Tenure']

    return df_cal

# Data transformation
df_model = Data_transformation_renaming(df_raw)

# Define feature columns and target column
cat_cols = ['Gender', 'InternetServiceType', 'PaymentMethod', 'ContractType']
binary_cols = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
               'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
               'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'InternetService', 'IsContracted']
num_cols = ['Tenure', 'MonthlyCharges', 'TotalCharges']
target_col = 'Churn'

# Create model features and target
X = df_model.drop(columns=[target_col])
y = df_model[target_col]

# Encode categorical variables
label_encoder = LabelEncoder()
for col in cat_cols:
    X[col] = label_encoder.fit_transform(X[col])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'churn_prediction_model.pkl')

print("Model trained and saved successfully!")
