from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt
import os

def train_and_evaluate_model(params, X_train, y_train, X_test, y_test, input_example):
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train['full_text'])
    X_test_tfidf  = vectorizer.transform(X_test['full_text'])

    # Inisialisasi model dengan parameter
    model = SVC(**params)
    model.fit(X_train_tfidf, y_train.to_numpy())
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)

    # Logging manual ke MLflow
    for k, v in params.items():
        mlflow.log_param(k, v)
    mlflow.log_metric("accuracy", accuracy)
    cr_dict = classification_report(y_test, y_pred, output_dict=True)
    cr_text = classification_report(y_test, y_pred)
    # log sebagai teks
    mlflow.log_text(cr_text, "classification_report.txt")
    # log metrik tambahan
    mlflow.log_metric("f1_macro", cr_dict["macro avg"]["f1-score"])
    mlflow.log_metric("f1_weighted", cr_dict["weighted avg"]["f1-score"])
    

    # Simpan confusion matrix ke file
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    # Log file matrix ke MLflow
    mlflow.log_artifact(cm_path)
    
    # Log model secara eksplisit
    mlflow.sklearn.log_model(sk_model=model, artifact_path="model", input_example=input_example)

    # Hapus file lokal agar rapi
    if os.path.exists(cm_path):
        os.remove(cm_path)

    return accuracy
