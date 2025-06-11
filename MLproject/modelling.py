import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import mlflow

def main(C, gamma, kernel, data_path):
    # mlflow.set_tracking_uri("https://yogaWidodo:bc00ae01433011469c5c110e4bfa6ec2d22d0641@dagshub.com/yogaWidodo/MLOP-Submission.mlflow")
    # mlflow.set_experiment("SVM_MODEL_SUBMISSION_MLOPS")

    tweets_df = pd.read_csv(data_path)
    tweets_df.dropna(subset=['full_text', 'sentiment'], inplace=True)

    X = tweets_df['full_text']
    y = tweets_df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    input_example = X_train.iloc[[0]]


    with mlflow.start_run():
        mlflow.log_param("C", C)
        mlflow.log_param("gamma", gamma)
        mlflow.log_param("kernel", kernel)

        model = SVC(C=C, gamma=gamma, kernel=kernel)
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
            )

        mlflow.log_metric("accuracy", accuracy)
        print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--gamma", type=str, default="scale")
    parser.add_argument("--kernel", type=str, default="rbf")
    parser.add_argument("--data_path", type=str, default="naturalisasi_dataset_cleaned.csv")
    args = parser.parse_args()

    # Convert gamma to float if it's numeric
    try:
        gamma = float(args.gamma)
    except ValueError:
        gamma = args.gamma  # "scale" or "auto"

    main(args.C, gamma, args.kernel, args.data_path)
