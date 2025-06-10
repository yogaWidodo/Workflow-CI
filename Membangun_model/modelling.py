from sklearn.model_selection import train_test_split, ParameterSampler
from modelling_tuning import train_and_evaluate_model
from sklearn.svm import SVC
import pandas as pd
import mlflow


# Set MLflow tracking dan eksperimen
mlflow.set_tracking_uri("https://yogaWidodo:bc00ae01433011469c5c110e4bfa6ec2d22d0641@dagshub.com/yogaWidodo/MLOP-Submission.mlflow")
mlflow.set_experiment("SVM_MODEL_SUBMISSION_MLOPS")

# Load dan siapkan data
tweets_df = pd.read_csv("Membangun_model/naturalisasi_dataset_cleaned.csv")
tweets_df.dropna(subset=['full_text', 'sentiment'], inplace=True)

X = tweets_df[['full_text']]
y = tweets_df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Contoh input untuk autolog
input_example = X_train.iloc[[0]]

# Hyperparameter tuning
param_grid = {
    'C': [1, 10],
    'gamma': [1, 0.1],
    'kernel': ['rbf'],
}
n_iter_search = 4

# Mulai eksperimen utama
for params in ParameterSampler(param_grid, n_iter=n_iter_search):
    with mlflow.start_run(run_name="Random Search SVM"):
        print("Running with:", params)
        acc = train_and_evaluate_model(params, X_train, y_train, X_test, y_test, input_example)
        print("Acc:", acc)
