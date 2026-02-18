"""
Credit Risk ML — Model Training, Tuning & Evaluation
=====================================================
Trains and compares 4 models for loan default prediction:
1. Logistic Regression (baseline)
2. XGBoost
3. LightGBM
4. Sklearn Gradient Boosting (CatBoost alternative)

INCLUDES:
- Hyperparameter tuning (RandomizedSearchCV)
- Comprehensive evaluation metrics (AUC-ROC, F1, Precision-Recall)
- Feature importance analysis
- Model comparison report

WHY THESE 4 MODELS?
───────────────────
1. Logistic Regression: ALWAYS start here. It's interpretable,
   fast, and gives you a performance floor. If your XGBoost
   only beats LogReg by 1%, you have a feature problem, not
   a model problem.

2. XGBoost: The industry workhorse. Regularization built-in,
   handles missing values natively, excellent for tabular data.
   Most fintech companies use XGBoost in production.

3. LightGBM: Microsoft's challenger. Uses histogram-based splits
   (faster than XGBoost on large datasets). Leaf-wise growth
   can overfit on small data but shines on 100K+ rows.

4. Sklearn GradientBoosting: Scikit-learn's native implementation.
   More conservative defaults, integrates seamlessly with sklearn
   ecosystem. Great for comparison and reproducibility.
   
   NOTE: CatBoost (Yandex) is the ideal 4th model for credit risk
   due to its ordered boosting and native categorical handling.
   It requires Python ≤3.12 for prebuilt wheels. In production,
   you'd use CatBoost instead.

Interview Insight: "Why not just use XGBoost?"
→ "Different gradient boosting implementations have different
   inductive biases. XGBoost uses level-wise growth (more conservative),
   LightGBM uses leaf-wise (faster but riskier). Comparing them ensures
   we pick the best one for our specific data distribution. In production,
   I'd also include CatBoost for its ordered boosting which reduces
   target leakage in the boosting process."
"""

import numpy as np
import pandas as pd
import json
import time
import joblib
import warnings
from pathlib import Path
from datetime import datetime

# Sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, classification_report, confusion_matrix,
    average_precision_score, roc_curve
)

# Gradient Boosting libraries
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

# ─── Paths ──────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. DATA LOADER
# ============================================================

def load_processed_data():
    """Load preprocessed train/test splits."""
    print("  Loading preprocessed data...")
    
    X_train = pd.read_csv(PROCESSED_DIR / 'X_train.csv')
    X_test = pd.read_csv(PROCESSED_DIR / 'X_test.csv')
    y_train = pd.read_csv(PROCESSED_DIR / 'y_train.csv').squeeze()
    y_test = pd.read_csv(PROCESSED_DIR / 'y_test.csv').squeeze()
    X_train_scaled = pd.read_csv(PROCESSED_DIR / 'X_train_scaled.csv')
    X_test_scaled = pd.read_csv(PROCESSED_DIR / 'X_test_scaled.csv')
    
    print(f"    -> Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"    -> Default rate: {y_train.mean():.1%} (train), {y_test.mean():.1%} (test)")
    
    return {
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'X_train_scaled': X_train_scaled, 'X_test_scaled': X_test_scaled,
    }


# ============================================================
# 2. MODEL DEFINITIONS + HYPERPARAMETER SPACES
# ============================================================

def get_model_configs():
    """
    Define models and their hyperparameter search spaces.
    
    WHY THESE HYPERPARAMETERS?
    ──────────────────────────
    Each parameter controls a specific aspect of the bias-variance tradeoff.
    
    Interview Insight: Don't just say "I tuned learning_rate and max_depth."
    Say WHY each parameter matters for credit risk specifically.
    """
    
    configs = {}
    
    # ─── 1. LOGISTIC REGRESSION (Baseline) ───
    # WHY LogReg?
    # - Interpretable: coefficients directly show feature importance
    # - Fast: trains in seconds, serves in microseconds
    # - Regulatory: some banks REQUIRE interpretable models for compliance
    # - Benchmark: if boosting barely beats LogReg, your features are weak
    configs['Logistic_Regression'] = {
        'model': LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver='saga',           # Best for large datasets
            class_weight='balanced'  # Handles class imbalance
        ),
        'params': {
            'C': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
            # WHY C? Controls regularization strength.
            # Low C = strong regularization (underfits)
            # High C = weak regularization (overfits)
            # In credit risk, moderate C (0.1-1.0) usually wins.
            
            'penalty': ['l1', 'l2'],
            # WHY L1 vs L2?
            # L1 (Lasso): drives weak features to zero → feature selection
            # L2 (Ridge): shrinks all features → better when all features matter
            # Credit risk: L1 often wins because many features are correlated
        },
        'use_scaled': True,  # LogReg NEEDS scaled features
        'n_iter': 14,
    }
    
    # ─── 2. XGBOOST ───
    # WHY XGBoost?
    # - Built-in L1/L2 regularization (gamma, lambda, alpha)
    # - Handles missing values natively
    # - Column subsampling reduces overfitting
    # - Most deployed ML model in fintech globally
    configs['XGBoost'] = {
        'model': xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            random_state=42,
            n_jobs=-1,
            tree_method='hist',   # Fastest method
            verbosity=0,
        ),
        'params': {
            'n_estimators': [100, 200, 300, 500],
            # WHY? More trees = more capacity. But with early stopping,
            # we'd use 1000+ in production. Here we cap for speed.
            
            'max_depth': [3, 4, 5, 6, 7],
            # WHY? Controls tree complexity. Shallow trees (3-4) = high bias.
            # Deep trees (7+) = overfit risk. Credit data usually: 4-6.
            
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            # WHY? Lower LR needs more trees but generalizes better.
            # Production rule: LR=0.01-0.05 with 500-1000 trees.
            
            'subsample': [0.7, 0.8, 0.9, 1.0],
            # WHY? Row sampling per tree. <1.0 adds randomness → reduces overfit.
            
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
            # WHY? Column sampling per tree. Decorrelates trees.
            
            'min_child_weight': [1, 3, 5, 7],
            # WHY? Minimum sum of instance weight in a child.
            # Higher = more conservative (fewer splits).
            
            'gamma': [0, 0.1, 0.2, 0.5],
            # WHY? Minimum loss reduction for a split. Acts as pruning.
            
            'reg_alpha': [0, 0.01, 0.1],
            # L1 regularization
            
            'reg_lambda': [1, 1.5, 2.0],
            # L2 regularization
            
            'scale_pos_weight': [1, 1.5],
            # WHY? Handles class imbalance by upweighting minority class.
        },
        'use_scaled': False,  # Trees don't need scaling
        'n_iter': 50,
    }
    
    # ─── 3. LIGHTGBM ───
    # WHY LightGBM?
    # - Histogram-based: bins continuous features → much faster
    # - Leaf-wise growth: finds the split with max delta loss
    # - GOSS (Gradient-based One-Side Sampling): faster on large data
    configs['LightGBM'] = {
        'model': lgb.LGBMClassifier(
            objective='binary',
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            is_unbalance=True,  # Handles imbalance
        ),
        'params': {
            'n_estimators': [100, 200, 300, 500],
            
            'max_depth': [3, 5, 7, -1],  # -1 = no limit (leaf-wise)
            # WHY -1? LightGBM leaf-wise growth benefits from unlimited depth
            # but controlled by num_leaves instead.
            
            'num_leaves': [15, 31, 50, 80],
            # WHY? THE key parameter for LightGBM. Controls complexity.
            # Rule of thumb: num_leaves < 2^max_depth to prevent overfitting.
            
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            
            'subsample': [0.7, 0.8, 0.9, 1.0],
            
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
            
            'min_child_samples': [5, 10, 20, 50],
            # WHY? Minimum data in a leaf. Prevents learning noise.
            
            'reg_alpha': [0, 0.01, 0.1, 1.0],
            'reg_lambda': [0, 0.01, 0.1, 1.0],
        },
        'use_scaled': False,
        'n_iter': 50,
    }
    
    # ─── 4. SKLEARN GRADIENT BOOSTING ───
    # WHY Sklearn GBM?
    # - Native sklearn integration (pipelines, cross-validation)
    # - More conservative defaults (less prone to overfitting)
    # - Deterministic and fully reproducible
    # - Good baseline for comparing against XGBoost/LightGBM
    #
    # NOTE: In production with Python ≤3.12, replace this with CatBoost:
    # - Ordered boosting reduces target leakage
    # - Native categorical handling (no OHE needed)
    # - Symmetric trees → more robust predictions
    configs['Sklearn_GBM'] = {
        'model': GradientBoostingClassifier(
            random_state=42,
            validation_fraction=0.1,
            n_iter_no_change=10,    # Early stopping
        ),
        'params': {
            'n_estimators': [100, 200, 300, 500],
            
            'max_depth': [3, 4, 5, 6],
            # WHY? Sklearn GBM uses depth-wise growth. 3-5 is standard.
            
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            
            'subsample': [0.7, 0.8, 0.9, 1.0],
            # WHY? Stochastic gradient boosting — Friedman (2002).
            
            'min_samples_split': [5, 10, 20],
            # WHY? Minimum samples to split a node. Higher = less overfit.
            
            'min_samples_leaf': [3, 5, 10, 20],
            # WHY? Minimum samples in a leaf. Critical regularizer.
            
            'max_features': ['sqrt', 'log2', 0.8],
            # WHY? Feature subsampling per tree. sqrt is a good default.
        },
        'use_scaled': False,
        'n_iter': 40,
    }
    
    return configs


# ============================================================
# 3. MODEL TRAINER
# ============================================================

class ModelTrainer:
    """
    Trains, tunes, and evaluates models with comprehensive reporting.
    
    WHY RandomizedSearchCV INSTEAD OF GridSearchCV?
    ───────────────────────────────────────────────
    GridSearch tests ALL combinations. For XGBoost with our params:
    4×5×4×4×4×4×4×3×3×2 = 92,160 combos × 5 folds = 460,800 fits!
    
    RandomizedSearchCV samples N random combinations (we use 50).
    Research (Bergstra & Bengio, 2012) shows random search finds
    near-optimal params in ~60% fewer iterations than grid search.
    
    In production, use Bayesian optimization (Optuna, Hyperopt)
    which is 3-5x more efficient. For a portfolio project,
    RandomizedSearchCV is the right balance of rigor and speed.
    """
    
    def __init__(self, cv_folds: int = 5, scoring: str = 'roc_auc'):
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.results = {}
        self.best_models = {}
        
        # WHY AUC-ROC AS SCORING METRIC?
        # - Accuracy is misleading with class imbalance
        # - AUC-ROC is threshold-independent
        # - Industry standard for credit risk scorecards
    
    def train_model(self, name: str, config: dict, data: dict) -> dict:
        """Train a single model with hyperparameter tuning."""
        print(f"\n  {'='*55}")
        print(f"  Training: {name}")
        print(f"  {'='*55}")
        
        start_time = time.time()
        
        # Select scaled or unscaled features
        if config['use_scaled']:
            X_train = data['X_train_scaled']
            X_test = data['X_test_scaled']
        else:
            X_train = data['X_train']
            X_test = data['X_test']
        
        y_train = data['y_train']
        y_test = data['y_test']
        
        # Setup cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        # Hyperparameter tuning
        print(f"  Tuning {config['n_iter']} random combos x {self.cv_folds}-fold CV...")
        
        search = RandomizedSearchCV(
            estimator=config['model'],
            param_distributions=config['params'],
            n_iter=config['n_iter'],
            scoring=self.scoring,
            cv=cv,
            random_state=42,
            n_jobs=-1,
            verbose=0,
            return_train_score=True,
        )
        
        search.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        # Best model
        best_model = search.best_estimator_
        best_params = search.best_params_
        cv_score = search.best_score_
        
        print(f"  Best CV {self.scoring}: {cv_score:.4f}")
        print(f"  Best Params: {best_params}")
        
        # Test set predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Comprehensive metrics
        metrics = self._compute_metrics(y_test, y_pred, y_pred_proba)
        metrics['cv_auc'] = round(cv_score, 4)
        metrics['train_time_seconds'] = round(train_time, 1)
        
        # Feature importance
        importance = self._get_feature_importance(best_model, X_train.columns.tolist(), name)
        
        # Store results
        result = {
            'model': best_model,
            'best_params': best_params,
            'metrics': metrics,
            'feature_importance': importance,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
        }
        
        self.results[name] = result
        self.best_models[name] = best_model
        
        self._print_metrics(name, metrics)
        
        return result
    
    def _compute_metrics(self, y_true, y_pred, y_pred_proba) -> dict:
        """
        Compute comprehensive evaluation metrics.
        
        WHY MULTIPLE METRICS?
        ─────────────────────
        No single metric tells the full story for credit risk:
        
        - AUC-ROC: Overall discriminative ability (threshold-independent)
        - AUC-PR: Focus on positive class (better for imbalanced data)
        - Precision: Of predicted defaults, how many actually defaulted?
          → High precision = fewer false alarms → less wasted investigation
        - Recall: Of actual defaults, how many did we catch?
          → High recall = fewer missed defaults → less financial loss
        - F1: Harmonic mean of Precision & Recall
        - KS Statistic: Maximum separation between TPR and FPR curves
          → Industry standard; KS > 0.40 is considered good
        - Gini: = 2 × AUC - 1. Another industry standard.
          → Gini > 0.50 is good, > 0.60 is excellent
        
        Interview Insight: "Which metric would you optimize for?"
        → "It depends on business cost. If false negatives (missed defaults)
           cost ₹5L each and false positives (rejected good customers) cost
           ₹50K in lost revenue, I'd optimize for recall with a precision floor."
        """
        auc_roc = roc_auc_score(y_true, y_pred_proba)
        auc_pr = average_precision_score(y_true, y_pred_proba)
        
        # KS Statistic
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        ks_stat = max(tpr - fpr)
        
        # Gini coefficient
        gini = 2 * auc_roc - 1
        
        return {
            'accuracy': round(accuracy_score(y_true, y_pred), 4),
            'precision': round(precision_score(y_true, y_pred), 4),
            'recall': round(recall_score(y_true, y_pred), 4),
            'f1_score': round(f1_score(y_true, y_pred), 4),
            'auc_roc': round(auc_roc, 4),
            'auc_pr': round(auc_pr, 4),
            'ks_statistic': round(ks_stat, 4),
            'gini': round(gini, 4),
        }
    
    def _get_feature_importance(self, model, feature_names: list, 
                                 model_name: str) -> pd.DataFrame:
        """Extract feature importance from the trained model."""
        
        if model_name == 'Logistic_Regression':
            importance = np.abs(model.coef_[0])
        elif hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            return pd.DataFrame()
        
        fi_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).reset_index(drop=True)
        
        fi_df['importance_normalized'] = fi_df['importance'] / fi_df['importance'].sum()
        
        return fi_df
    
    def _print_metrics(self, name: str, metrics: dict):
        """Pretty-print model metrics."""
        print(f"\n  {'-'*50}")
        print(f"  {name} -- Test Set Results")
        print(f"  {'-'*50}")
        print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}  {'***' if metrics['auc_roc'] > 0.85 else '**' if metrics['auc_roc'] > 0.75 else '*'}")
        print(f"  AUC-PR:      {metrics['auc_pr']:.4f}")
        print(f"  F1 Score:    {metrics['f1_score']:.4f}")
        print(f"  Precision:   {metrics['precision']:.4f}")
        print(f"  Recall:      {metrics['recall']:.4f}")
        print(f"  Accuracy:    {metrics['accuracy']:.4f}")
        print(f"  KS Stat:     {metrics['ks_statistic']:.4f}  {'(Good)' if metrics['ks_statistic'] > 0.40 else '(Moderate)'}")
        print(f"  Gini:        {metrics['gini']:.4f}")
        print(f"  CV AUC:      {metrics['cv_auc']:.4f}")
        print(f"  Train Time:  {metrics['train_time_seconds']:.1f}s")
    
    def train_all(self, data: dict) -> dict:
        """Train all models and return comparison."""
        configs = get_model_configs()
        
        for name, config in configs.items():
            self.train_model(name, config, data)
        
        return self.results
    
    def get_comparison_table(self) -> pd.DataFrame:
        """Create a comparison table of all trained models."""
        rows = []
        for name, result in self.results.items():
            row = {'Model': name}
            row.update(result['metrics'])
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.sort_values('auc_roc', ascending=False).reset_index(drop=True)
        return df
    
    def print_comparison(self):
        """Print formatted model comparison."""
        comp = self.get_comparison_table()
        
        print(f"\n{'='*90}")
        print(f"  MODEL COMPARISON -- SORTED BY AUC-ROC")
        print(f"{'='*90}")
        print(f"\n  {'Model':<25} {'AUC-ROC':>8} {'AUC-PR':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'KS':>8} {'Gini':>8} {'Time':>8}")
        print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        
        for idx, row in comp.iterrows():
            marker = ' <BEST>' if idx == 0 else ''
            print(f"  {row['Model']:<25} {row['auc_roc']:>8.4f} {row['auc_pr']:>8.4f} "
                  f"{row['f1_score']:>8.4f} {row['precision']:>8.4f} {row['recall']:>8.4f} "
                  f"{row['ks_statistic']:>8.4f} {row['gini']:>8.4f} {row['train_time_seconds']:>7.1f}s{marker}")
        
        best = comp.iloc[0]
        print(f"\n  >> Best model: {best['Model']} (AUC-ROC: {best['auc_roc']:.4f})")
        
        # Top features from best model
        best_name = best['Model']
        best_fi = self.results[best_name]['feature_importance']
        if not best_fi.empty:
            print(f"\n  Top 15 Features ({best_name}):")
            print(f"  {'-'*55}")
            for i, row in best_fi.head(15).iterrows():
                bar = '#' * int(row['importance_normalized'] * 50)
                print(f"    {i+1:2d}. {row['feature']:<40} {bar} {row['importance_normalized']:.3f}")
        
        print(f"\n{'='*90}")
    
    def save_results(self):
        """Save all results to disk."""
        print(f"\n  Saving results...")
        
        # Comparison table
        comp = self.get_comparison_table()
        comp.to_csv(RESULTS_DIR / 'model_comparison.csv', index=False)
        
        # Feature importance per model
        for name, result in self.results.items():
            if not result['feature_importance'].empty:
                result['feature_importance'].to_csv(
                    RESULTS_DIR / f'{name}_feature_importance.csv', index=False
                )
        
        # Save trained models
        for name, model in self.best_models.items():
            model_path = MODELS_DIR / f'{name}_model.pkl'
            joblib.dump(model, model_path)
        
        # Detailed JSON report
        report = {
            'timestamp': datetime.now().isoformat(),
            'comparison': comp.to_dict(orient='records'),
            'best_model': comp.iloc[0]['Model'],
            'best_auc_roc': float(comp.iloc[0]['auc_roc']),
            'model_details': {}
        }
        
        for name, result in self.results.items():
            report['model_details'][name] = {
                'best_params': {k: (v.tolist() if isinstance(v, np.ndarray) 
                                   else float(v) if isinstance(v, (np.floating, np.integer))
                                   else v) 
                               for k, v in result['best_params'].items()},
                'metrics': result['metrics'],
                'top_10_features': result['feature_importance'].head(10)[
                    ['feature', 'importance_normalized']
                ].to_dict(orient='records') if not result['feature_importance'].empty else [],
            }
        
        with open(RESULTS_DIR / 'training_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Classification reports + confusion matrices
        y_test = pd.read_csv(PROCESSED_DIR / 'y_test.csv').squeeze()
        
        for name, result in self.results.items():
            report_path = RESULTS_DIR / f'{name}_classification_report.txt'
            report_text = classification_report(y_test, result['y_pred'],
                                                target_names=['Non-Default', 'Default'])
            cm = confusion_matrix(y_test, result['y_pred'])
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"{'='*50}\n")
                f.write(f"Classification Report: {name}\n")
                f.write(f"{'='*50}\n\n")
                f.write(report_text)
                f.write(f"\nConfusion Matrix:\n")
                f.write(f"              Predicted\n")
                f.write(f"              Non-Def  Default\n")
                f.write(f"Actual Non-Def  {cm[0][0]:>5}    {cm[0][1]:>5}\n")
                f.write(f"Actual Default  {cm[1][0]:>5}    {cm[1][1]:>5}\n")
                f.write(f"\nInterpretation:\n")
                f.write(f"  True Negatives:  {cm[0][0]:>5} (correctly identified non-default)\n")
                f.write(f"  False Positives: {cm[0][1]:>5} (false alarm -> rejected good customer)\n")
                f.write(f"  False Negatives: {cm[1][0]:>5} (MISSED DEFAULT -> financial loss!)\n")
                f.write(f"  True Positives:  {cm[1][1]:>5} (correctly caught default)\n")
        
        print(f"  [OK] Comparison:   {RESULTS_DIR / 'model_comparison.csv'}")
        print(f"  [OK] Report:       {RESULTS_DIR / 'training_report.json'}")
        print(f"  [OK] Models saved: {MODELS_DIR}")
        print(f"  [OK] Feature importance + classification reports saved")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("="*65)
    print("  CREDIT RISK ML -- MODEL TRAINING & COMPARISON")
    print("  LogReg  |  XGBoost  |  LightGBM  |  Sklearn GBM")
    print("="*65)
    
    # Load data
    data = load_processed_data()
    
    # Train all models
    trainer = ModelTrainer(cv_folds=5, scoring='roc_auc')
    trainer.train_all(data)
    
    # Print comparison
    trainer.print_comparison()
    
    # Save everything
    trainer.save_results()
    
    print(f"\n  [OK] All models trained, tuned, and evaluated!")
    print(f"  [OK] Results: {RESULTS_DIR}")
    print(f"  [OK] Models:  {MODELS_DIR}")


if __name__ == '__main__':
    main()
