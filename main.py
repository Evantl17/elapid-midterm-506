# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# load the data
print("Loading data...")
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# merge the data
print("Merging test data with training data...")
test_data_merged = test_data.merge(train_data.drop(columns='Score'), on='Id', how='left')

# preprocess text
print("Preprocessing text data...")
train_data['Summary'] = train_data['Summary'].fillna("")
train_data['Text'] = train_data['Text'].fillna("")
train_data['Combined_Text'] = train_data['Summary'] + " " + train_data['Text']

test_data_merged['Summary'] = test_data_merged['Summary'].fillna("")
test_data_merged['Text'] = test_data_merged['Text'].fillna("")
test_data_merged['Combined_Text'] = test_data_merged['Summary'] + " " + test_data_merged['Text']
train_data = train_data.dropna(subset=['Score'])

# define features and set the target
X = train_data['Combined_Text']
y = train_data['Score'].astype(int)
# split data
print("Splitting data...")
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create the pipeline with TF-IDF and logistic regression
print("Setting up pipeline...")
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=500, stop_words='english')),  # Reduced max_features
    ('classifier', LogisticRegression(max_iter=1000, random_state=42, solver='saga'))  # Faster solver
])

# grid search
param_grid = {
    'tfidf__max_features': [500, 1000],  # Fewer values to test
    'classifier__C': [0.1, 1]  # Fewer values to test
}

#run gridsearch
print("Starting GridSearchCV with fewer parameters and folds...")
grid_search = GridSearchCV(pipeline, param_grid, cv=2, scoring='accuracy', n_jobs=-1, verbose=1)  # Reduced CV folds
grid_search.fit(X_train, Y_train)
print("Grid search complete.")

# find the best model 
print("Evaluating the best model...")
best_model = grid_search.best_estimator_
Y_test_predictions = best_model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_test_predictions)
classification_report_text = classification_report(Y_test, Y_test_predictions)

print("Best Model Accuracy:", accuracy)
print("Classification Report:\n", classification_report_text)

# use best model to find the final prediction
print("Generating final predictions on test data...")
test_bow_features = test_data_merged['Combined_Text']
test_predictions = best_model.predict(test_bow_features)

# make submission file
print("Saving submission file...")
submission = pd.DataFrame({'Id': test_data_merged['Id'], 'Score': test_predictions})
submission.to_csv('submission.csv', index=False)

print("Execution complete!")
