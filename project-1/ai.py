import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import os
from nltk.corpus import stopwords
import string
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # You can change this to your actual number of cores

def load_data(file_path):
    df = pd.read_csv(file_path)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    return df

def remove_news_source(text):
    if not isinstance(text, str) or text == '':
        return text
    
    # Find the last dash and remove everything after it
    last_dash_pos = text.rfind(' - ')
    if (last_dash_pos != -1):
        return text[:last_dash_pos].strip()
    return text.strip()

def preprocess_data(df):
    df['label'] = df['label'].map({'up': 1, 'down': 0})
    text_columns = [
        'tech_1', 'tech_2', 'tech_3', 'tech_4',
        'business_1', 'business_2', 'business_3', 'business_4',
        'market_1', 'market_2', 'market_3', 'market_4',
        'economy_1', 'economy_2', 'economy_3', 'economy_4',
        'events_1', 'events_2', 'events_3', 'events_4'
    ]
    
    # Clean each text column by removing news sources
    for column in text_columns:
        df[column] = df[column].fillna('').apply(remove_news_source)
        df[column] = df[column].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
        
    df['combined_text'] = df[text_columns].fillna('').agg(' '.join, axis=1)
    
    # Clean stopwords, punctuation, and convert to lowercase
    df['combined_text'] = df['combined_text'].str.lower()
    df['combined_text'] = df['combined_text'].str.replace(r'[^\w\s]', '')
    
    
    df_majority = df[df['label'] == 1]
    df_minority = df[df['label'] == 0]
    n_samples = min(len(df_majority), len(df_minority))
    df_majority_downsampled = resample(df_majority, replace=False, n_samples=n_samples, random_state=42)
    df_minority_upsampled = resample(df_minority, replace=False, n_samples=n_samples, random_state=42)
    df_balanced = pd.concat([df_majority_downsampled, df_minority_upsampled])
    
    return df_balanced[['date', 'combined_text', 'label']]

def extract_features(text_data, type):
    if type == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=500)
    else:
        vectorizer = CountVectorizer(ngram_range=(2,2))
    features = vectorizer.fit_transform(text_data)
    return features, vectorizer

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train, y_train)
    return model

def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    print(f"{model_name} Accuracy:", accuracy_score(y_test, y_pred))
    print(f"{model_name} Classification Report:\n", classification_report(y_test, y_pred))

def perform_cross_validation(model, X, y, model_name, n_folds=5):
    """
    Perform k-fold cross-validation and report results
    
    Args:
        model: The model to evaluate
        X: Feature matrix
        y: Target variable
        model_name: Name of the model for reporting
        n_folds: Number of folds for cross-validation
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    
    print(f"\n{model_name} Cross-Validation Results:")
    print(f"Fold accuracies: {cv_scores}")
    print(f"Mean accuracy: {cv_scores.mean():.4f}")
    print(f"Standard deviation: {cv_scores.std():.4f}")
    
    return cv_scores.mean()

def apply_kmeans(X, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(X)

def plot_kmeans_clusters(X, labels):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray())
    
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="Set2", legend="full")
    plt.title("K-Means Clustering Result")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Cluster")
    plt.show()

def apply_dbscan(X, eps=0.6, min_samples = 8):
    """Use DBSCAN for news clustering"""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    clusters = dbscan.fit_predict(X)
    return clusters

def plot_dbscan_clusters(X, labels):
    """Use PCA for dimensionality reduction, visualize DBSCAN clustering results"""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray()) 

    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette="Set2", legend="full")
    plt.title("DBSCAN Clustering Result")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(title="Cluster")
    plt.show()

def main():
    file_path = './dataset/news_with_stock_labels.csv'
    df = load_data(file_path)
    df = preprocess_data(df)
    
    ''' experiment with random sampling
    random_pick_count = input("Please enter the number of random samples you want to pick: ")
    random_pick_count = int(random_pick_count)
    
    df = df.sample(n=random_pick_count, random_state=42)
    '''
    X, vectorizer = extract_features(df['combined_text'], type='tfidf')
    y = df['label']
    
    # Original train-test split for individual evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define models
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    svm_model = SVC(kernel='linear', random_state=42)
    
    # Perform cross-validation
    print("\n===== Performing 5-Fold Cross-Validation =====")
    lr_cv_score = perform_cross_validation(lr_model, X, y, "LogisticRegression")
    rf_cv_score = perform_cross_validation(rf_model, X, y, "RandomForest")
    svm_cv_score = perform_cross_validation(svm_model, X, y, "SVM")
    
    # Train models on entire training set and evaluate on test set
    print("\n===== Training and Evaluating on Train-Test Split =====")
    print("Training Logistic Regression...")
    lr_model.fit(X_train, y_train)
    evaluate_model(lr_model, X_test, y_test, "LogisticRegression")
    
    print("Training Random Forest...")
    rf_model.fit(X_train, y_train)
    evaluate_model(rf_model, X_test, y_test, "RandomForest")
    
    print("Training SVM...")
    svm_model.fit(X_train, y_train)
    evaluate_model(svm_model, X_test, y_test, "SVM") 
    
    # Determine best model based on cross-validation scores
    best_model_name = {lr_cv_score: "Logistic Regression", 
                       rf_cv_score: "Random Forest", 
                       svm_cv_score: "SVM"}[max(lr_cv_score, rf_cv_score, svm_cv_score)]
    print(f"\nBest performing model based on cross-validation: {best_model_name}")
    
    print("Running K-Means clustering...")
    df['kmeans_cluster'] = apply_kmeans(X, n_clusters=2)
    print(df[['combined_text', 'kmeans_cluster']].head(10))
    plot_kmeans_clusters(X, df['kmeans_cluster'])
    
    
    print("Running DBSCAN clustering...")
    df['dbscan_cluster'] = apply_dbscan(X)
    print(df[['combined_text', 'dbscan_cluster']].head(10))
    plot_dbscan_clusters(X, df['dbscan_cluster'])
    

# Example usage for the remove_news_source function
if __name__ == "__main__":
    
    main()
