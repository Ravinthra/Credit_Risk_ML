"""
Credit Risk ML -- Inference-Time Preprocessor
==============================================
Transforms raw form inputs into the 66-feature vector expected by the model.

WHY A SEPARATE PREPROCESSOR FOR INFERENCE?
- Training preprocessing (src/preprocessing.py) operates on full datasets
  with fit/transform patterns (learn stats from train, apply to test).
- Inference preprocessing operates on a SINGLE row from user input.
- We hardcode the learned imputation values and encoding maps from
  the training pipeline (stored in pipeline_report.json).
- This avoids needing to load the entire training pipeline at serve time.

Interview Insight:
"How do you handle preprocessing at inference time?"
-> "I separate training preprocessing (fit_transform on batches) from
   inference preprocessing (transform-only on single rows). The learned
   statistics (medians, encodings) are serialized and loaded at serve time.
   This prevents training-serving skew -- the #1 cause of silent model
   degradation in production."
"""

import numpy as np
import pandas as pd


class InferencePreprocessor:
    """
    Converts raw user inputs into the 66-feature model input vector.

    The user provides ~15 raw fields. This preprocessor:
    1. Computes derived features (ratios, logs, buckets)
    2. Applies ordinal encoding for ordered categories
    3. Creates one-hot encoded columns
    4. Sets missing indicators
    
    NOTE: Repayment-derived features (late_payment_ratio, avg_dpd, etc.)
    and label-leaking features (past_defaults, historical_default_rate)
    have been removed from the model to prevent data leakage.
    """

    # ── Ordinal Encoding Maps (learned from training) ────────
    # WHY hardcoded? These mappings were learned during training.
    # In production, you'd serialize these with the model pipeline.
    EDUCATION_MAP = {
        'Below Secondary': 0, 'Secondary': 1, 'Higher Secondary': 2,
        'Bachelor': 3, 'Master': 4, 'PhD': 5,
    }

    CREDIT_SCORE_TIER_MAP = {
        'Poor': 0, 'Fair': 1, 'Good': 2, 'Very Good': 3, 'Excellent': 4,
    }

    AGE_GROUP_MAP = {
        'Young': 0, 'Adult': 1, 'Middle-Aged': 2, 'Senior': 3,
    }

    UTILIZATION_BUCKET_MAP = {
        'Low': 0, 'Moderate': 1, 'High': 2, 'Critical': 3,
    }

    TERM_BUCKET_MAP = {
        'Short': 0, 'Medium': 1, 'Long': 2,
    }

    RATE_TIER_MAP = {
        'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3,
    }

    # ── One-Hot Encoding Categories ──────────────────────────
    GENDER_CATS = ['Female', 'Male', 'Other']
    MARITAL_CATS = ['Divorced', 'Married', 'Single', 'Widowed']
    EMPLOYMENT_CATS = ['Business', 'Freelance', 'Salaried', 'Self-Employed', 'Unemployed']
    LOAN_TYPE_CATS = ['Auto', 'Business', 'Credit Card', 'Education', 'Gold', 'Home', 'Personal']

    # ── The exact 66 features in the order the model expects ─
    # NOTE: Repayment behavioral features and label-leaking features
    #       have been REMOVED to prevent data leakage.
    FEATURE_ORDER = [
        'age', 'age_group', 'dependents', 'education', 'annual_income',
        'log_annual_income', 'employment_stability_ratio', 'income_per_dependent',
        'city_tier', 'verification_score', 'credit_score', 'credit_score_tier',
        'total_accounts', 'active_accounts', 'active_account_ratio',
        'overdue_30_count', 'overdue_60_count', 'overdue_90_count',
        'delinquency_severity_score', 'has_serious_delinquency',
        'credit_utilization', 'utilization_bucket', 'enquiries_last_6m',
        'enquiry_velocity', 'credit_history_length_months', 'is_thin_file',
        'loan_amount', 'log_loan_amount', 'loan_term_months', 'term_bucket',
        'interest_rate', 'rate_tier', 'emi_amount', 'is_secured',
        'dti_ratio', 'emi_to_income_ratio', 'loan_to_income_ratio',
        'interest_to_principal_ratio',
        'total_loans', 'total_exposure_to_income', 'loan_type_diversity',
        # Missing indicators
        'education_is_missing', 'employment_stability_ratio_is_missing',
        'credit_score_is_missing', 'enquiries_last_6m_is_missing',
        'enquiry_velocity_is_missing', 'interest_rate_is_missing',
        # One-hot: gender
        'gender_Female', 'gender_Male', 'gender_Other',
        # One-hot: marital_status
        'marital_status_Divorced', 'marital_status_Married',
        'marital_status_Single', 'marital_status_Widowed',
        # One-hot: employment_type
        'employment_type_Business', 'employment_type_Freelance',
        'employment_type_Salaried', 'employment_type_Self-Employed',
        'employment_type_Unemployed',
        # One-hot: loan_type
        'loan_type_Auto', 'loan_type_Business', 'loan_type_Credit Card',
        'loan_type_Education', 'loan_type_Gold', 'loan_type_Home',
        'loan_type_Personal',
    ]

    def preprocess(self, form_data: dict) -> pd.DataFrame:
        """
        Transform raw form data into model-ready 66-feature DataFrame.

        Parameters
        ----------
        form_data : dict
            Raw user inputs from the Django form. Expected keys:
            age, gender, education, marital_status, dependents,
            employment_type, annual_income, credit_score, loan_amount,
            loan_type, loan_term_months, interest_rate, existing_loans,
            credit_utilization, employment_years

        Returns
        -------
        pd.DataFrame with shape (1, 66) matching FEATURE_ORDER
        """
        row = {}

        # ── 1. Direct mapping from form ──────────────────────
        age = float(form_data.get('age', 30))
        annual_income = float(form_data.get('annual_income', 500000))
        credit_score = float(form_data.get('credit_score', 650))
        loan_amount = float(form_data.get('loan_amount', 200000))
        loan_term = float(form_data.get('loan_term_months', 36))
        interest_rate = float(form_data.get('interest_rate', 12.0))
        dependents = int(form_data.get('dependents', 0))
        existing_loans = int(form_data.get('existing_loans', 1))
        credit_util = float(form_data.get('credit_utilization', 40.0))
        employment_years = float(form_data.get('employment_years', 3.0))

        # ── 2. Derived features (computed, not collected) ────
        monthly_income = annual_income / 12
        emi_amount = self._compute_emi(loan_amount, interest_rate, loan_term)
        log_annual_income = np.log1p(annual_income)
        log_loan_amount = np.log1p(loan_amount)

        # Ratios
        dti_ratio = (emi_amount / monthly_income) if monthly_income > 0 else 0
        emi_to_income_ratio = dti_ratio
        loan_to_income_ratio = loan_amount / annual_income if annual_income > 0 else 0
        interest_to_principal = (interest_rate * loan_term / 1200)
        income_per_dep = annual_income / (dependents + 1)
        employment_stability = min(employment_years / 10.0, 1.0)

        # Buckets / tiers
        age_group = self._bucket_age(age)
        credit_tier = self._bucket_credit_score(credit_score)
        util_bucket = self._bucket_utilization(credit_util)
        term_bucket = self._bucket_term(loan_term)
        rate_tier = self._bucket_rate(interest_rate)
        is_thin_file = 1 if credit_score < 600 and existing_loans <= 1 else 0

        # Loan properties
        is_secured = 1 if form_data.get('loan_type', 'Personal') in ['Home', 'Auto', 'Gold'] else 0

        # ── 3. Build feature values ────────────────────────────
        # NOTE: Repayment behavioral features (late_payment_ratio, avg_dpd)
        # and label-leaking features (past_defaults, historical_default_rate)
        # have been REMOVED from the model to prevent data leakage.
        row['age'] = age
        row['age_group'] = self.AGE_GROUP_MAP.get(age_group, 1)
        row['dependents'] = dependents
        row['education'] = self.EDUCATION_MAP.get(form_data.get('education', 'Bachelor'), 3)
        row['annual_income'] = annual_income
        row['log_annual_income'] = log_annual_income
        row['employment_stability_ratio'] = employment_stability
        row['income_per_dependent'] = income_per_dep
        row['city_tier'] = int(form_data.get('city_tier', 2))
        row['verification_score'] = int(form_data.get('verification_score', 3))
        row['credit_score'] = credit_score
        row['credit_score_tier'] = self.CREDIT_SCORE_TIER_MAP.get(credit_tier, 2)
        row['total_accounts'] = existing_loans + 2  # Assume some savings/current accounts
        row['active_accounts'] = existing_loans + 1
        row['active_account_ratio'] = row['active_accounts'] / max(row['total_accounts'], 1)
        row['overdue_30_count'] = 0
        row['overdue_60_count'] = 0
        row['overdue_90_count'] = 0
        row['delinquency_severity_score'] = 0
        row['has_serious_delinquency'] = 0
        row['credit_utilization'] = credit_util
        row['utilization_bucket'] = self.UTILIZATION_BUCKET_MAP.get(util_bucket, 1)
        row['enquiries_last_6m'] = 1
        row['enquiry_velocity'] = 0.5
        row['credit_history_length_months'] = int(employment_years * 12)
        row['is_thin_file'] = is_thin_file
        row['loan_amount'] = loan_amount
        row['log_loan_amount'] = log_loan_amount
        row['loan_term_months'] = loan_term
        row['term_bucket'] = self.TERM_BUCKET_MAP.get(term_bucket, 1)
        row['interest_rate'] = interest_rate
        row['rate_tier'] = self.RATE_TIER_MAP.get(rate_tier, 1)
        row['emi_amount'] = emi_amount
        row['is_secured'] = is_secured
        row['dti_ratio'] = dti_ratio
        row['emi_to_income_ratio'] = emi_to_income_ratio
        row['loan_to_income_ratio'] = loan_to_income_ratio
        row['interest_to_principal_ratio'] = interest_to_principal
        row['total_loans'] = existing_loans
        row['total_exposure_to_income'] = (loan_amount + existing_loans * 100000) / annual_income if annual_income > 0 else 0
        row['loan_type_diversity'] = min(existing_loans, 3)

        # ── 4. Missing indicators (all 0 since form provides values) ──
        row['education_is_missing'] = 0
        row['employment_stability_ratio_is_missing'] = 0
        row['credit_score_is_missing'] = 0
        row['enquiries_last_6m_is_missing'] = 0
        row['enquiry_velocity_is_missing'] = 0
        row['interest_rate_is_missing'] = 0

        # ── 5. One-hot encoding ──────────────────────────────
        # Gender
        gender = form_data.get('gender', 'Male')
        for cat in self.GENDER_CATS:
            row[f'gender_{cat}'] = 1 if gender == cat else 0

        # Marital status
        marital = form_data.get('marital_status', 'Single')
        for cat in self.MARITAL_CATS:
            row[f'marital_status_{cat}'] = 1 if marital == cat else 0

        # Employment type
        emp_type = form_data.get('employment_type', 'Salaried')
        for cat in self.EMPLOYMENT_CATS:
            row[f'employment_type_{cat}'] = 1 if emp_type == cat else 0

        # Loan type
        loan_type = form_data.get('loan_type', 'Personal')
        for cat in self.LOAN_TYPE_CATS:
            row[f'loan_type_{cat}'] = 1 if loan_type == cat else 0

        # ── 6. Build DataFrame in exact feature order ────────
        df = pd.DataFrame([row])[self.FEATURE_ORDER]
        return df

    # ── Helper Methods ───────────────────────────────────────

    @staticmethod
    def _compute_emi(principal, annual_rate, term_months):
        """
        Compute EMI using standard amortization formula.
        EMI = P * r * (1+r)^n / ((1+r)^n - 1)
        """
        if annual_rate <= 0 or term_months <= 0:
            return principal / max(term_months, 1)
        r = annual_rate / 12 / 100  # Monthly rate
        n = term_months
        emi = principal * r * (1 + r)**n / ((1 + r)**n - 1)
        return round(emi, 2)

    @staticmethod
    def _bucket_age(age):
        if age < 25: return 'Young'
        elif age < 35: return 'Adult'
        elif age < 50: return 'Middle-Aged'
        else: return 'Senior'

    @staticmethod
    def _bucket_credit_score(score):
        if score < 580: return 'Poor'
        elif score < 670: return 'Fair'
        elif score < 740: return 'Good'
        elif score < 800: return 'Very Good'
        else: return 'Excellent'

    @staticmethod
    def _bucket_utilization(util):
        if util < 30: return 'Low'
        elif util < 50: return 'Moderate'
        elif util < 75: return 'High'
        else: return 'Critical'

    @staticmethod
    def _bucket_term(months):
        if months <= 12: return 'Short'
        elif months <= 36: return 'Medium'
        else: return 'Long'

    @staticmethod
    def _bucket_rate(rate):
        if rate < 10: return 'Low'
        elif rate < 15: return 'Medium'
        elif rate < 20: return 'High'
        else: return 'Very High'
