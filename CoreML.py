import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
 
# First, add all necessary imports at the top
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
    VotingClassifier,
    BaggingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from itertools import product  # Add this import
import joblib  # Add this for model saving
import sys
import warnings
import os
warnings.filterwarnings('ignore')

# Rest of the code remains exactly the same...
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    StackingClassifier,
    VotingClassifier,
    BaggingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import (
    LogisticRegression,
    RidgeClassifier,
    SGDClassifier
)
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import warnings

"""
, "SVM": (
            SVC(probability=True, class_weight=class_weight_dict, random_state=42),
            {
                "C": [0.1, 1.0, 10.0],
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto"]
            }
        ),

"""
warnings.filterwarnings('ignore')
def calculate_class_weights(y_train):
    """Calculate class weights from training data"""
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    weight_dict = dict(zip(classes, weights))
    class_ratio = weights[1] / weights[0] if len(weights) > 1 else 1.0
    return weight_dict, class_ratio

def get_models_and_params(class_weight_dict, class_ratio):
    """Define all models and their hyperparameter grids"""
    models = {
        "Decision Tree": (
            DecisionTreeClassifier(class_weight=class_weight_dict, random_state=42),
            {
                "max_depth": [3, 5, 7, 10, 15, 20],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "criterion": ["gini", "entropy"]
            }
        ),
         "Random Forest": (
            RandomForestClassifier(class_weight=class_weight_dict, random_state=42),
            {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, 15],
                "min_samples_split": [2, 5],
                "max_features": ["sqrt", "log2"]
            }
        ),
        "Extra Trees": (
            ExtraTreesClassifier(class_weight=class_weight_dict, random_state=42),
            {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, 15],
                "min_samples_split": [2, 5]
            }
        ),
       

        "KNN": (
            KNeighborsClassifier(weights='distance'),
            {
                "n_neighbors": [3, 5, 7],
                "metric": ["euclidean", "manhattan"]
            }
        ),
        "Naive Bayes": (
            GaussianNB(priors=[1/(1+class_ratio), class_ratio/(1+class_ratio)]),
            {
                "var_smoothing": [1e-9, 1e-8, 1e-7]
            }
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 1.0]
            }
        ),
        "AdaBoost": (
            AdaBoostClassifier(
                DecisionTreeClassifier(class_weight=class_weight_dict),
                random_state=42
            ),
            {
                "n_estimators": [50, 100],
                "learning_rate": [0.01, 0.1]
            }
        ),
        "XGBoost": (
            XGBClassifier(
                scale_pos_weight=class_ratio,
                random_state=42
            ),
            {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.01, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]
            }
        ),
        "LightGBM": (
            LGBMClassifier(
                scale_pos_weight=class_ratio,
                random_state=42
            ),
            {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.01, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]
            }
        ),
        "CatBoost": (
            CatBoostClassifier(
                random_state=42,
                verbose=False,
                scale_pos_weight=class_ratio
            ),
            {
                "iterations": [100, 200],
                "learning_rate": [0.01, 0.1],
                "depth": [4, 6],
                "l2_leaf_reg": [1, 3]
            }
        ),
        "Neural Network": (
            MLPClassifier(
                random_state=42,
                max_iter=1000
            ),
            {
                "hidden_layer_sizes": [(50,), (100,)],
                "alpha": [0.0001, 0.001],
                "learning_rate": ["adaptive"]
            }
        ),
        "Logistic Regression": (
            LogisticRegression(
                class_weight=class_weight_dict,
                random_state=42,
                max_iter=1000
            ),
            {
                "C": [0.1, 1.0],
                "penalty": ["l2"],
                "solver": ["liblinear"]
            }
        )       
        
    }
    return models

def create_ensemble_models(base_models, class_weight_dict, class_ratio):
    """Create voting and stacking ensembles from base models"""
    
    # Create base classifiers with class weights
    rf_clf = RandomForestClassifier(
        class_weight=class_weight_dict,
        random_state=42,
        n_estimators=100
    )
    
    xgb_clf = XGBClassifier(
        scale_pos_weight=class_ratio,
        random_state=42,
        n_estimators=100
    )
    
    lgbm_clf = LGBMClassifier(
        scale_pos_weight=class_ratio,
        random_state=42,
        n_estimators=10,
        
    )
    
    # Create voting classifier
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', rf_clf),
            ('xgb', xgb_clf),
            ('lgbm', lgbm_clf)
        ],
        voting='soft'
    )
    
    # Create stacking classifier
    stacking_clf = StackingClassifier(
        estimators=[
            ('rf', rf_clf),
            ('xgb', xgb_clf),
            ('lgbm', lgbm_clf)
        ],
        final_estimator=LogisticRegression(
            class_weight=class_weight_dict,
            random_state=42
        ),
        stack_method='predict_proba'
    )
    
    ensemble_models = {
        "Voting Classifier": (
            voting_clf,
            {
                "weights": [[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2]]
            }
        ),
        "Stacking Classifier": (
            stacking_clf,
            {
                "final_estimator__C": [0.1, 1.0]
            }
        )
    }
    
    return ensemble_models

def custom_grid_search(model_name, estimator, param_grid, X_train, y_train, X_val, y_val, X_test, y_test):
    """
    Perform grid search using validation set for evaluation instead of CV.
    Returns performance metrics for all parameter combinations.
    """
    all_results = []
    best_score = -np.inf
    best_params = None
    best_model = None
    
    # Convert param_grid to list of dictionaries if it isn't already
    if isinstance(param_grid, dict):
        param_grid = [param_grid]
    
    # Generate all parameter combinations
    param_combinations = []
    for params in param_grid:
        keys = params.keys()
        values = params.values()
        for instance in product(*values):
            param_combinations.append(dict(zip(keys, instance)))
    
    for params in param_combinations:
        if params:
            try:
                # Create and fit model with current parameters
                model = clone(estimator)
                model.set_params(**params)
                
                # Fit on training data
                model.fit(X_train, y_train)
                
                # Get predictions for all sets
                train_pred = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_train)
                val_pred = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_val)
                test_pred = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
                
                # Calculate scores
                train_score = roc_auc_score(y_train, train_pred)
                val_score = roc_auc_score(y_val, val_pred)
                test_score = roc_auc_score(y_test, test_pred)
                
                # Store results for this parameter combination
                result = {
                    'Model': model_name,
                    'ML_Parameters': str(params),
                    'Train AUC': train_score,
                    'Validation AUC': val_score,
                    'Test AUC': test_score
                }
                all_results.append(result)
                
                # Update best model if current one is better
                if val_score > best_score:
                    best_score = val_score
                    best_params = params
                    best_model = model
                    
                #print(f"Model: {model_name}")
                #print(f"Parameters: {params}")
                #print(f"Train AUC: {train_score:.4f}")
                #print(f"Validation AUC: {val_score:.4f}")
                #print(f"Test AUC: {test_score:.4f}\n")
                
            except Exception as e:
                print(f"Error with {model_name} using parameters {params}: {str(e)}")

                # Add failed attempt to results with NaN scores
                result = {
                    'Model': model_name,
                    'ML_Parameters': str(params),
                    'Train AUC': 0,
                    'Validation AUC': 0,
                    'Test AUC':0
                }
                all_results.append(result)
                continue
    
    return best_model, best_params, best_score, pd.DataFrame(all_results)

def train_and_evaluate_machine_learning(X_train, X_test, X_val, y_train, y_test, y_val):
    """Train models and evaluate their performance"""
    
    # Calculate class weights
    class_weight_dict, class_ratio = calculate_class_weights(y_train)
    
    # Get models and their parameters
    models = get_models_and_params(class_weight_dict, class_ratio)
    ensemble_models = create_ensemble_models(models, class_weight_dict, class_ratio)
    all_models = {**models}
    all_models = {**models, **ensemble_models}
    

    all_results_dfs = []
    
    for model_name, (model, param_grid) in all_models.items():
        print(f"\nTraining {model_name}...")
        
        try:
            # Perform grid search with separate validation set
            best_model, best_params, val_score, results_df = custom_grid_search(
                model_name=model_name,
                estimator=model,
                param_grid=param_grid,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test
            )
            
            # Append results
            if not results_df.empty:
                all_results_dfs.append(results_df)
            
            # Save the best model if one was found
            if best_model is not None:
                model_filename = f"best_models/{model_name.lower().replace(' ', '_')}.pkl"
                os.makedirs("best_models", exist_ok=True)
                joblib.dump(best_model, model_filename)
                
                # Print best results
                best_row = results_df.loc[results_df['Validation AUC'].idxmax()]
                print(f"\n{model_name} Best Results:")
                print(f"Best Parameters: {best_row['ML_Parameters']}")
                print(f"Best Train AUC: {best_row['Train AUC']:.4f}")
                print(f"Best Validation AUC: {best_row['Validation AUC']:.4f}")
                print(f"Test AUC: {best_row['Test AUC']:.4f}")


                best_result = pd.DataFrame([{
                    'Model': model_name,
                    'Parameters': best_row['ML_Parameters'],
                    'Train AUC': best_row['Train AUC'],
                    'Validation AUC': best_row['Validation AUC'],
                    'Test AUC': best_row['Test AUC']
                }])

                # Do not include the best result for each model as they contain overfitting information for specific fold.
                # all_results_dfs.append(best_result)
        
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            # Add a dummy result for failed model
            failed_result = pd.DataFrame([{
                'Model': model_name,
                'Parameters': str(param_grid),
                'Train AUC': 0,
                'Validation AUC': 0,
                'Test AUC': 0
            }])
            all_results_dfs.append(failed_result)
            continue

    # Combine all results
    if not all_results_dfs:
        return pd.DataFrame(columns=['Model', 'Parameters', 'Train AUC', 'Validation AUC', 'Test AUC'])
    
    final_results = pd.concat(all_results_dfs, ignore_index=True)
    
    # Sort by Model name and Validation AUC (descending)
    # final_results = final_results.sort_values(['Model', 'Validation AUC'], ascending=[True, False])
    
    return final_results