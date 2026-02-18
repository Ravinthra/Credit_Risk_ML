"""
Credit Risk ML — Preprocessing Pipeline
========================================
Advanced preprocessing for credit risk modeling.

MODULE STRUCTURE:
─────────────────
This file contains 3 main components:
1. MissingValueHandler — MNAR-aware imputation
2. FeatureEncoder — Categorical encoding strategies
3. FeatureScaler — Intelligent scaling
4. PreprocessingPipeline — Orchestrates everything

WHY MODULAR DESIGN?
───────────────────
In production ML, preprocessing is the #1 source of bugs.
Modular design allows:
- Unit testing each component independently
- Swapping strategies without rewriting the pipeline
- Reproducing EXACT same transforms on new data (inference)
- Clear debugging when predictions drift in production

Interview Insight: "Walk me through your preprocessing pipeline"
→ Don't say "I used StandardScaler and LabelEncoder."
→ Say "I built a modular pipeline with MNAR-aware imputation that
   preserves missingness signals, ordinal encoding for ordered
   categories, and RobustScaler for outlier-sensitive features.
   Each component is fit on training data and applied to test data
   to prevent data leakage."
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
import warnings
import json
from pathlib import Path

warnings.filterwarnings('ignore')

# ─── Paths ──────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. MISSING VALUE HANDLER
# ============================================================

class MissingValueHandler:
    """
    MNAR-Aware Missing Value Handler.
    
    WHY NOT JUST SimpleImputer?
    ──────────────────────────
    SimpleImputer assumes MCAR (Missing Completely At Random).
    Our data has MNAR (Missing Not At Random):
    
    - Missing credit_score → thin-file customer (no history)
    - Missing years_employed → self-employed (no formal tenure)
    - Missing enquiries → bureau didn't return this field
    
    STRATEGY:
    1. Create binary "is_missing" indicator columns FIRST
       → The fact that data is missing IS a feature
    2. Then impute with median (numeric) or mode (categorical)
    3. For credit_score: use -1 flag (domain convention)
    
    Interview Insight: "Why indicator columns?"
    → "Missingness itself carries signal. A missing credit score
       means the customer is new-to-credit, which is a risk factor.
       If I just impute with median, I LOSE that information."
    """
    
    def __init__(self):
        self.numeric_medians = {}
        self.categorical_modes = {}
        self.missing_columns = []
        self._fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'MissingValueHandler':
        """
        Learn imputation values from training data ONLY.
        
        WHY FIT ON TRAIN ONLY?
        → Using test data statistics for imputation = data leakage.
        → In production, you don't HAVE future data to compute stats from.
        → This is the #1 preprocessing mistake in Kaggle notebooks.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
        
        # Find columns with missing values
        cols_with_missing = df.columns[df.isna().any()].tolist()
        self.missing_columns = cols_with_missing
        
        # Store medians for numeric columns
        for col in numeric_cols:
            if col in cols_with_missing:
                self.numeric_medians[col] = df[col].median()
        
        # Store modes for categorical columns
        for col in categorical_cols:
            if col in cols_with_missing:
                mode_val = df[col].mode()
                self.categorical_modes[col] = mode_val[0] if len(mode_val) > 0 else 'Unknown'
        
        self._fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply imputation using learned values.
        
        Steps:
        1. Add binary missing indicators (before imputing)
        2. Impute numeric with median
        3. Impute categorical with mode
        4. Special handling for credit_score (domain-specific)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()!")
        
        df = df.copy()
        
        # Step 1: Create missingness indicators
        # WHY BEFORE IMPUTING? Once we impute, we can't tell what was missing.
        for col in self.missing_columns:
            if df[col].isna().any():
                df[f'{col}_is_missing'] = df[col].isna().astype(int)
        
        # Step 2: Impute numeric columns
        for col, median_val in self.numeric_medians.items():
            if col in df.columns:
                df[col] = df[col].fillna(median_val)
        
        # Step 3: Impute categorical columns
        for col, mode_val in self.categorical_modes.items():
            if col in df.columns:
                df[col] = df[col].fillna(mode_val)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
    
    def get_report(self) -> dict:
        """Return imputation summary for documentation."""
        return {
            'numeric_imputed': {k: round(v, 2) for k, v in self.numeric_medians.items()},
            'categorical_imputed': self.categorical_modes,
            'indicator_columns_created': [f'{c}_is_missing' for c in self.missing_columns]
        }


# ============================================================
# 2. FEATURE ENCODER
# ============================================================

class FeatureEncoder:
    """
    Intelligent Feature Encoder for Credit Risk Data.
    
    ENCODING STRATEGY (WHY each choice):
    ─────────────────────────────────────
    
    1. ORDINAL ENCODING — for features with natural order
       - education: High School < Bachelor < Master < PhD
       - credit_score_tier: Very_Poor < Poor < Fair < Good < Excellent
       - age_group: Young < Early_Career < Mid_Career < ...
       WHY? Preserves the monotonic relationship with risk.
       One-hot encoding would LOSE this ordering information.
    
    2. ONE-HOT ENCODING — for nominal (no order) categories
       - gender: Male/Female/Other — no natural ordering
       - loan_type: Personal/Home/Auto — no inherent order
       - employment_type: Salaried/Self-Employed/... — no clear order
       WHY? OHE treats each category independently, which is correct
       when there's no ordinal relationship.
    
    3. TARGET ENCODING — for high-cardinality categories (if any)
       Not used here (our categories are low-cardinality), but
       would be the right choice for zip_code, employer_name, etc.
       WHY? Captures the relationship between category and target
       without exploding dimensionality like OHE would.
    
    Interview Insight:
    "Why not just LabelEncoder for everything?"
    → "LabelEncoder assigns arbitrary integers (Male=0, Female=1).
       Tree models can handle this, but linear models interpret it
       as Female > Male, which is meaningless. For linear models,
       we need OHE for nominal and ordinal for ordered categories."
    """
    
    # Define encoding strategies per feature
    ORDINAL_MAPPINGS = {
        'education': ['Other', 'High School', 'Bachelor', 'Master', 'PhD'],
        'credit_score_tier': ['Very_Poor', 'Poor', 'Fair', 'Good', 'Excellent'],
        'age_group': ['Young', 'Early_Career', 'Mid_Career', 'Pre_Retirement', 'Senior'],
        'utilization_bucket': ['Low', 'Moderate', 'High', 'Critical'],
        'term_bucket': ['Short', 'Medium', 'Long', 'Very_Long'],
        'rate_tier': ['Prime', 'Near_Prime', 'Subprime', 'Deep_Subprime'],
    }
    
    ONEHOT_COLUMNS = [
        'gender', 'marital_status', 'employment_type', 'loan_type'
    ]
    
    def __init__(self):
        self.ordinal_encoders = {}
        self.onehot_columns_created = []
        self._fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'FeatureEncoder':
        """Learn encoding mappings from training data."""
        
        # Fit ordinal encoders
        for col, categories in self.ORDINAL_MAPPINGS.items():
            if col in df.columns:
                self.ordinal_encoders[col] = {cat: idx for idx, cat in enumerate(categories)}
        
        # For one-hot, we just need to know which columns exist
        # (we'll handle unseen categories gracefully)
        self._onehot_categories = {}
        for col in self.ONEHOT_COLUMNS:
            if col in df.columns:
                self._onehot_categories[col] = sorted(df[col].dropna().unique().tolist())
        
        self._fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply encoding transformations."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()!")
        
        df = df.copy()
        
        # Step 1: Ordinal encoding
        for col, mapping in self.ordinal_encoders.items():
            if col in df.columns:
                # Map known values, assign -1 to unknowns
                df[col] = df[col].map(mapping).fillna(-1).astype(int)
        
        # Step 2: One-hot encoding
        for col in self.ONEHOT_COLUMNS:
            if col in df.columns:
                # Get dummies with prefix
                dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
                
                # Ensure all categories from training are present
                for cat in self._onehot_categories.get(col, []):
                    col_name = f"{col}_{cat}"
                    if col_name not in dummies.columns:
                        dummies[col_name] = 0
                
                self.onehot_columns_created.extend(dummies.columns.tolist())
                
                # Drop original column and add dummies
                df = df.drop(columns=[col])
                df = pd.concat([df, dummies], axis=1)
        
        # Step 3: Convert boolean columns to int
        bool_cols = df.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            df[col] = df[col].astype(int)
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


# ============================================================
# 3. FEATURE SCALER
# ============================================================

class FeatureScaler:
    """
    Intelligent Feature Scaler.
    
    WHY NOT SCALE EVERYTHING THE SAME WAY?
    ───────────────────────────────────────
    
    Different features need different scaling:
    
    1. STANDARD SCALING (z-score) — for normally distributed features
       - age, credit_score, loan_term_months
       - These are roughly Gaussian after our transformations
    
    2. ROBUST SCALING — for features with outliers
       - annual_income, loan_amount (heavily right-skewed even after log)
       - Uses median/IQR instead of mean/std, so outliers don't dominate
    
    3. NO SCALING — for already-bounded features
       - Binary flags (0/1), ratios (0-1), ordinal encodings
       - Scaling these adds no value and hurts interpretability
    
    Interview Insight:
    "Why RobustScaler for income?"
    → "Income has extreme outliers (₹3L vs ₹1Cr). StandardScaler uses
       mean/std which are sensitive to outliers. RobustScaler uses
       median and IQR, so the CEO's ₹5Cr salary doesn't distort the
       scaling for the 99% earning under ₹30L."
    
    "Does scaling matter for tree models?"
    → "No! Trees split on thresholds, so scaling is irrelevant.
       But we also train Logistic Regression as a baseline, which
       IS sensitive to scale. So we scale for LR and use unscaled
       for tree models. This is a production best practice."
    """
    
    # Features that should NOT be scaled
    SKIP_SCALING = {
        # Binary features
        'is_default', 'is_secured', 'has_serious_delinquency',
        'ever_90_plus_dpd', 'ever_60_plus_dpd', 'is_thin_file',
        'phone_verified', 'email_verified',
        # Already bounded ratios (0-1)
        'late_payment_ratio', 'partial_payment_ratio',
        'auto_debit_ratio', 'active_account_ratio',
        'avg_payment_to_due_ratio', 'credit_utilization',
        'dti_ratio', 'ltv_ratio',
        # Encoded ordinal features
        'education', 'credit_score_tier', 'age_group',
        'utilization_bucket', 'term_bucket', 'rate_tier',
        'verification_score', 'city_tier',
        # ID columns
        'loan_id', 'customer_id',
    }
    
    # Features needing RobustScaler (outlier-prone)
    ROBUST_SCALE_FEATURES = {
        'annual_income', 'loan_amount', 'emi_amount',
        'income_per_dependent', 'total_loan_exposure',
        'avg_loan_amount', 'total_amount_paid',
        'total_interest_payable', 'collateral_value',
    }
    
    def __init__(self):
        self.standard_scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.standard_cols = []
        self.robust_cols = []
        self._fitted = False
    
    def fit(self, df: pd.DataFrame) -> 'FeatureScaler':
        """Learn scaling parameters from training data."""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Classify columns
        self.robust_cols = [c for c in numeric_cols 
                          if c in self.ROBUST_SCALE_FEATURES and c in df.columns]
        self.standard_cols = [c for c in numeric_cols 
                            if c not in self.SKIP_SCALING 
                            and c not in self.ROBUST_SCALE_FEATURES
                            and not c.endswith('_is_missing')  # Don't scale indicators
                            and not c.startswith(('gender_', 'marital_status_', 
                                                  'employment_type_', 'loan_type_'))]
        
        # Fit scalers
        if self.standard_cols:
            self.standard_scaler.fit(df[self.standard_cols].values)
        if self.robust_cols:
            self.robust_scaler.fit(df[self.robust_cols].values)
        
        self._fitted = True
        return self
    
    def transform(self, df: pd.DataFrame, apply_scaling: bool = True) -> pd.DataFrame:
        """
        Apply scaling transformations.
        
        Parameters
        ----------
        apply_scaling : bool
            If False, returns data without scaling (for tree models).
            If True, applies Standard/Robust scaling (for linear models).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()!")
        
        df = df.copy()
        
        if not apply_scaling:
            return df
        
        # Apply StandardScaler
        if self.standard_cols:
            existing_std = [c for c in self.standard_cols if c in df.columns]
            if existing_std:
                # Need to handle potential column mismatch
                temp = pd.DataFrame(
                    self.standard_scaler.transform(df[self.standard_cols].values),
                    columns=self.standard_cols,
                    index=df.index
                )
                df[self.standard_cols] = temp
        
        # Apply RobustScaler
        if self.robust_cols:
            existing_rob = [c for c in self.robust_cols if c in df.columns]
            if existing_rob:
                temp = pd.DataFrame(
                    self.robust_scaler.transform(df[self.robust_cols].values),
                    columns=self.robust_cols,
                    index=df.index
                )
                df[self.robust_cols] = temp
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, apply_scaling: bool = True) -> pd.DataFrame:
        return self.fit(df).transform(df, apply_scaling)


# ============================================================
# 4. PREPROCESSING PIPELINE (Orchestrator)
# ============================================================

class PreprocessingPipeline:
    """
    End-to-end preprocessing pipeline for credit risk modeling.
    
    Orchestrates:
    1. Train/test split (stratified on target)
    2. Missing value handling (MNAR-aware)
    3. Feature encoding (ordinal + one-hot)
    4. Feature scaling (Standard + Robust)
    5. Produces TWO versions:
       - Scaled (for Logistic Regression)
       - Unscaled (for tree-based models: XGBoost, LightGBM, CatBoost)
    
    WHY TWO VERSIONS?
    ─────────────────
    - Logistic Regression needs scaled features (gradient descent)
    - Tree models don't need scaling (split-based, scale-invariant)
    - Using scaled data for trees won't break anything, but unscaled
      is slightly better (no information loss from rounding)
    
    WHY STRATIFIED SPLIT?
    ─────────────────────
    Our default rate is ~40%. Without stratification, we could get
    a test set with 50% defaults and train with 35%, making evaluation
    metrics unreliable. Stratification ensures equal class proportions.
    """
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.missing_handler = MissingValueHandler()
        self.encoder = FeatureEncoder()
        self.scaler = FeatureScaler()
        self.feature_names = None
        self.target_col = 'is_default'
        self.id_col = 'loan_id'
    
    def run(self, df: pd.DataFrame) -> dict:
        """
        Execute full preprocessing pipeline.
        
        Returns dict with:
        - X_train, X_test, y_train, y_test (unscaled — for trees)
        - X_train_scaled, X_test_scaled (scaled — for LogReg)
        - feature_names: list of final feature names
        - pipeline_report: dict with preprocessing decisions
        """
        print("="*60)
        print("  PREPROCESSING PIPELINE")
        print("="*60)
        
        # ─── Step 0: Separate target and ID ───
        print("\n  [1/6] Separating target and features...")
        y = df[self.target_col].astype(int)
        drop_cols = [self.target_col, self.id_col]
        # Also drop any string columns that aren't in our encoding list
        # (like loan_purpose which is free text)
        extra_drops = []
        for col in df.select_dtypes(include='object').columns:
            if col not in (self.encoder.ONEHOT_COLUMNS + 
                          list(self.encoder.ORDINAL_MAPPINGS.keys())):
                extra_drops.append(col)
        
        drop_cols.extend(extra_drops)
        X = df.drop(columns=[c for c in drop_cols if c in df.columns])
        
        print(f"    → Features: {X.shape[1]}, Samples: {X.shape[0]}")
        print(f"    → Dropped non-feature columns: {drop_cols + extra_drops}")
        
        # ─── Step 1: Train/Test Split (BEFORE preprocessing!) ───
        # WHY SPLIT FIRST?
        # If we impute/encode on the full dataset, information from
        # the test set leaks into preprocessing parameters.
        # This is the CORRECT order: Split → Fit(train) → Transform(both)
        print("\n  [2/6] Stratified train/test split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y  # Maintain class proportions
        )
        print(f"    → Train: {X_train.shape[0]} samples ({y_train.mean():.1%} default)")
        print(f"    → Test:  {X_test.shape[0]} samples ({y_test.mean():.1%} default)")
        
        # ─── Step 2: Handle Missing Values ───
        print("\n  [3/6] Handling missing values (MNAR-aware)...")
        X_train = self.missing_handler.fit_transform(X_train)
        X_test = self.missing_handler.transform(X_test)
        
        missing_report = self.missing_handler.get_report()
        n_indicators = len(missing_report['indicator_columns_created'])
        print(f"    → Created {n_indicators} missingness indicator columns")
        print(f"    → Imputed {len(missing_report['numeric_imputed'])} numeric columns (median)")
        print(f"    → Imputed {len(missing_report['categorical_imputed'])} categorical columns (mode)")
        print(f"    → Remaining NaN: train={X_train.isna().sum().sum()}, test={X_test.isna().sum().sum()}")
        
        # ─── Step 3: Encode Categorical Features ───
        print("\n  [4/6] Encoding categorical features...")
        X_train = self.encoder.fit_transform(X_train)
        X_test = self.encoder.transform(X_test)
        
        print(f"    → Ordinal encoded: {list(self.encoder.ORDINAL_MAPPINGS.keys())}")
        print(f"    → One-hot encoded: {self.encoder.ONEHOT_COLUMNS}")
        print(f"    → Features after encoding: {X_train.shape[1]}")
        
        # ─── Step 4: Final cleanup ───
        print("\n  [5/6] Final cleanup...")
        
        # Ensure all columns are numeric
        for col in X_train.columns:
            if X_train[col].dtype == 'object':
                print(f"    ⚠ Converting {col} to numeric")
                X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
                X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)
        
        # Align columns (test might have different OHE columns)
        X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
        
        self.feature_names = X_train.columns.tolist()
        print(f"    → Final feature count: {len(self.feature_names)}")
        
        # ─── Step 5: Scale (separate version for linear models) ───
        print("\n  [6/6] Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train, apply_scaling=True)
        X_test_scaled = self.scaler.transform(X_test, apply_scaling=True)
        
        print(f"    → Standard-scaled: {len(self.scaler.standard_cols)} features")
        print(f"    → Robust-scaled:   {len(self.scaler.robust_cols)} features")
        print(f"    → Unscaled:        {X_train.shape[1] - len(self.scaler.standard_cols) - len(self.scaler.robust_cols)} features")
        
        # ─── Summary ───
        print("\n" + "="*60)
        print("  PIPELINE SUMMARY")
        print("="*60)
        print(f"  Input:              {df.shape[0]} rows × {df.shape[1]} cols")
        print(f"  Output (train):     {X_train.shape[0]} rows × {X_train.shape[1]} cols")
        print(f"  Output (test):      {X_test.shape[0]} rows × {X_test.shape[1]} cols")
        print(f"  Target balance:     {y_train.mean():.1%} default (train)")
        print(f"  Missing indicators: {n_indicators} columns added")
        print(f"  Encoding:           {len(self.encoder.ORDINAL_MAPPINGS)} ordinal + {len(self.encoder.ONEHOT_COLUMNS)} one-hot")
        print("="*60)
        
        return {
            # Unscaled (for tree models)
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            # Scaled (for linear models)
            'X_train_scaled': X_train_scaled,
            'X_test_scaled': X_test_scaled,
            # Metadata
            'feature_names': self.feature_names,
            'pipeline_report': {
                'missing_handling': missing_report,
                'ordinal_encoded': list(self.encoder.ORDINAL_MAPPINGS.keys()),
                'onehot_encoded': self.encoder.ONEHOT_COLUMNS,
                'standard_scaled': self.scaler.standard_cols,
                'robust_scaled': self.scaler.robust_cols,
                'train_size': X_train.shape[0],
                'test_size': X_test.shape[0],
                'n_features': len(self.feature_names),
                'default_rate_train': float(y_train.mean()),
                'default_rate_test': float(y_test.mean()),
            }
        }


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Run the full preprocessing pipeline."""
    
    # Load ML features from Step 1
    input_path = DATA_DIR / "ml_features.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"ML features not found at {input_path}. "
            "Run database/build_features.py first!"
        )
    
    print(f"  Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Loaded: {df.shape[0]} rows × {df.shape[1]} columns\n")
    
    # Run pipeline
    pipeline = PreprocessingPipeline(test_size=0.2, random_state=42)
    result = pipeline.run(df)
    
    # Save processed data
    print(f"\n  Saving processed data...")
    result['X_train'].to_csv(PROCESSED_DIR / 'X_train.csv', index=False)
    result['X_test'].to_csv(PROCESSED_DIR / 'X_test.csv', index=False)
    result['y_train'].to_csv(PROCESSED_DIR / 'y_train.csv', index=False)
    result['y_test'].to_csv(PROCESSED_DIR / 'y_test.csv', index=False)
    result['X_train_scaled'].to_csv(PROCESSED_DIR / 'X_train_scaled.csv', index=False)
    result['X_test_scaled'].to_csv(PROCESSED_DIR / 'X_test_scaled.csv', index=False)
    
    # Save pipeline report
    report_path = PROCESSED_DIR / 'pipeline_report.json'
    with open(report_path, 'w') as f:
        json.dump(result['pipeline_report'], f, indent=2, default=str)
    
    print(f"  ✓ Processed data saved to: {PROCESSED_DIR}")
    print(f"  ✓ Pipeline report saved to: {report_path}")
    
    # Print feature list for reference
    print(f"\n  Final Feature List ({len(result['feature_names'])} features):")
    print(f"  {'─'*50}")
    for i, fname in enumerate(result['feature_names'], 1):
        print(f"    {i:3d}. {fname}")
    
    print(f"\n  ✓ Preprocessing complete! Ready for model training.")


if __name__ == '__main__':
    main()
