import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from lime.lime_tabular import LimeTabularExplainer

def load_data(file_path):
    df = pd.read_excel(file_path)
    X = df.drop(columns=["Label"]).values
    y = df["Label"].values
    return X, y, list(df.drop(columns=["Label"]).columns)

def create_pipeline(model_type='classification'):
    base_models = {
        'classification': [
            ('dt', DecisionTreeClassifier()),
            ('knn', KNeighborsClassifier()),
            ('svm', SVC(probability=True))
        ],
        'regression': [
            ('dt', DecisionTreeRegressor()),
            ('knn', KNeighborsRegressor()),
            ('svm', SVR())
        ]
    }

    meta_models = {
        'classification': LogisticRegression(),
        'regression': Ridge()
    }

    stacking_model = StackingClassifier(estimators=base_models['classification'], final_estimator=meta_models['classification']) \
        if model_type == 'classification' else \
        StackingRegressor(estimators=base_models['regression'], final_estimator=meta_models['regression'])

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('stacking', stacking_model)
    ])

    return pipeline

def train_and_evaluate(pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred_pipeline = pipeline.predict(X_test)
    model_type = 'classification' if isinstance(pipeline.named_steps['stacking'], StackingClassifier) else 'regression'

    print(f'Pipeline Score: {pipeline.score(X_test, y_test):.4f}')
    if model_type == 'classification':
        print("Classification Report (Pipeline):")
        print(classification_report(y_test, y_pred_pipeline))
    else:
        print("Regression Metrics (Pipeline):")
        print(f'MSE: {mean_squared_error(y_test, y_pred_pipeline):.4f}')
        print(f'R² Score: {r2_score(y_test, y_pred_pipeline):.4f}')

    stacking_model = pipeline.named_steps['stacking']
    scaler = pipeline.named_steps['scaler']
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    stacking_model.fit(X_train_scaled, y_train)
    y_pred_stacking = stacking_model.predict(X_test_scaled)

    print(f'Stacking Model Score: {stacking_model.score(X_test_scaled, y_test):.4f}')
    if model_type == 'classification':
        print("Classification Report (Stacking Classifier):")
        print(classification_report(y_test, y_pred_stacking))
    else:
        print("Regression Metrics (Stacking Regressor):")
        print(f'MSE: {mean_squared_error(y_test, y_pred_stacking):.4f}')
        print(f'R² Score: {r2_score(y_test, y_pred_stacking):.4f}')

    return pipeline

def explain_model(pipeline, X_train, X_test, y_train, y_test, feature_names):
    model_type = 'classification' if isinstance(pipeline.named_steps['stacking'], StackingClassifier) else 'regression'

    explainer = LimeTabularExplainer(
        X_train,
        mode=model_type,
        feature_names=feature_names,
        discretize_continuous=True
    )

    instance = X_test[0]
    true_label = y_test[0]

    prediction_function = (
        pipeline.predict_proba if model_type == 'classification' else pipeline.predict
    )

    prediction = prediction_function([instance])
    predicted_class = np.argmax(prediction[0]) if model_type == 'classification' else prediction[0]

    print("\nLIME Explanation for 1st Test Instance:")
    print(f"Actual Class: {true_label}")
    print(f"Predicted Class: {predicted_class}")
    if model_type == 'classification':
        print(f"Predicted Probabilities: {np.round(prediction[0], 4)}")

    exp = explainer.explain_instance(instance, prediction_function)

    try:
        exp.show_in_notebook()
    except:
        print("\nFeature Contributions (LIME):")
        print(exp.as_list())

file_path = "Judgment_Embeddings_InLegalBERT.xlsx"
X, y, feature_names = load_data(file_path)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = create_pipeline('classification')
trained_pipeline = train_and_evaluate(pipeline, X_train, X_test, y_train, y_test)

explain_model(trained_pipeline, X_train, X_test, y_train, y_test, feature_names)
