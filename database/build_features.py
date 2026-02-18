"""
Credit Risk ML — SQL Feature Engineering (SQLite Compatible)
============================================================
Executes SQL feature engineering queries on the SQLite database
and produces the final ML-ready dataset.

WHY THIS SCRIPT?
────────────────
The SQL in feature_engineering.sql is written for PostgreSQL (production).
This script adapts those queries for SQLite, which we use locally.
In production, you'd run the PostgreSQL queries directly.
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "credit_risk.db"
OUTPUT_PATH = DATA_DIR / "ml_features.csv"


def connect_db():
    """Connect to SQLite database."""
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. "
            "Run generate_data.py first!"
        )
    conn = sqlite3.connect(str(DB_PATH))
    return conn


def build_feature_table(conn) -> pd.DataFrame:
    """
    Execute SQL feature engineering and return ML-ready DataFrame.
    
    This mirrors the v_ml_feature_table view from feature_engineering.sql,
    adapted for SQLite syntax.
    """
    
    query = """
    WITH demographic_features AS (
        SELECT 
            customer_id,
            age,
            CASE 
                WHEN age BETWEEN 18 AND 25 THEN 'Young'
                WHEN age BETWEEN 26 AND 35 THEN 'Early_Career'
                WHEN age BETWEEN 36 AND 50 THEN 'Mid_Career'
                WHEN age BETWEEN 51 AND 60 THEN 'Pre_Retirement'
                ELSE 'Senior'
            END AS age_group,
            gender,
            marital_status,
            dependents,
            education,
            employment_type,
            annual_income,
            annual_income / 12.0 AS monthly_income,
            LN(MAX(annual_income, 1)) AS log_annual_income,
            CASE WHEN age > 0 THEN years_employed * 1.0 / age ELSE 0 END 
                AS employment_stability_ratio,
            CASE WHEN dependents > 0 THEN annual_income / dependents
                 ELSE annual_income END AS income_per_dependent,
            city_tier,
            CASE 
                WHEN phone_verified = 1 AND email_verified = 1 THEN 2
                WHEN phone_verified = 1 OR email_verified = 1 THEN 1
                ELSE 0
            END AS verification_score
        FROM customers
    ),
    
    bureau_features AS (
        SELECT 
            customer_id,
            credit_score,
            CASE 
                WHEN credit_score >= 750 THEN 'Excellent'
                WHEN credit_score >= 700 THEN 'Good'
                WHEN credit_score >= 650 THEN 'Fair'
                WHEN credit_score >= 550 THEN 'Poor'
                ELSE 'Very_Poor'
            END AS credit_score_tier,
            total_accounts,
            active_accounts,
            CASE WHEN total_accounts > 0 
                 THEN CAST(active_accounts AS REAL) / total_accounts
                 ELSE 0 END AS active_account_ratio,
            overdue_30_count,
            overdue_60_count,
            overdue_90_count,
            (overdue_30_count * 1 + overdue_60_count * 3 + overdue_90_count * 10) 
                AS delinquency_severity_score,
            CASE WHEN overdue_90_count > 0 THEN 1 ELSE 0 END AS has_serious_delinquency,
            CASE WHEN total_credit_limit > 0 
                 THEN MIN(current_balance / total_credit_limit, 1.0)
                 ELSE 0 END AS credit_utilization,
            CASE 
                WHEN total_credit_limit > 0 AND current_balance / total_credit_limit <= 0.30 THEN 'Low'
                WHEN total_credit_limit > 0 AND current_balance / total_credit_limit <= 0.60 THEN 'Moderate'
                WHEN total_credit_limit > 0 AND current_balance / total_credit_limit <= 0.80 THEN 'High'
                ELSE 'Critical'
            END AS utilization_bucket,
            enquiries_last_6m,
            enquiries_last_12m,
            CASE WHEN enquiries_last_12m > 0 
                 THEN CAST(enquiries_last_6m AS REAL) / enquiries_last_12m
                 ELSE 0 END AS enquiry_velocity,
            credit_history_length_months,
            CASE WHEN credit_history_length_months < 12 THEN 1 ELSE 0 END AS is_thin_file
        FROM credit_history
    ),
    
    loan_features AS (
        SELECT 
            la.loan_id,
            la.customer_id,
            la.loan_amount,
            LN(MAX(la.loan_amount, 1)) AS log_loan_amount,
            la.loan_term_months,
            CASE 
                WHEN la.loan_term_months <= 12 THEN 'Short'
                WHEN la.loan_term_months <= 60 THEN 'Medium'
                WHEN la.loan_term_months <= 120 THEN 'Long'
                ELSE 'Very_Long'
            END AS term_bucket,
            la.interest_rate,
            CASE 
                WHEN la.interest_rate <= 10 THEN 'Prime'
                WHEN la.interest_rate <= 15 THEN 'Near_Prime'
                WHEN la.interest_rate <= 20 THEN 'Subprime'
                ELSE 'Deep_Subprime'
            END AS rate_tier,
            la.emi_amount,
            la.loan_type,
            la.is_secured,
            la.dti_ratio,
            la.ltv_ratio,
            la.is_default,
            CASE WHEN c.annual_income / 12.0 > 0 
                 THEN la.emi_amount / (c.annual_income / 12.0)
                 ELSE NULL END AS emi_to_income_ratio,
            CASE WHEN c.annual_income > 0 
                 THEN la.loan_amount / c.annual_income
                 ELSE NULL END AS loan_to_income_ratio,
            (la.emi_amount * la.loan_term_months) - la.loan_amount AS total_interest_payable,
            CASE WHEN la.loan_amount > 0 
                 THEN ((la.emi_amount * la.loan_term_months) - la.loan_amount) / la.loan_amount
                 ELSE 0 END AS interest_to_principal_ratio
        FROM loan_applications la
        JOIN customers c ON la.customer_id = c.customer_id
    ),
    
    -- NOTE: repayment_features CTE REMOVED
    -- Repayment behavioral features (late_payment_ratio, avg_dpd, max_dpd)
    -- are observed AFTER the loan is disbursed and thus leak future information.
    -- In production, at application time we do NOT know how the borrower will
    -- behave on THIS loan. Including these features causes AUC inflation.
    
    customer_agg AS (
        SELECT 
            c.customer_id,
            COUNT(la.loan_id) AS total_loans,
            -- past_defaults and historical_default_rate REMOVED
            -- These directly leak the target label back as a feature,
            -- especially when a customer has only one loan.
            SUM(la.loan_amount) AS total_loan_exposure,
            CASE WHEN c.annual_income > 0 
                 THEN SUM(la.loan_amount) / c.annual_income
                 ELSE NULL END AS total_exposure_to_income,
            ROUND(AVG(la.loan_amount), 2) AS avg_loan_amount,
            COUNT(DISTINCT la.loan_type) AS loan_type_diversity
        FROM customers c
        LEFT JOIN loan_applications la ON c.customer_id = la.customer_id
        GROUP BY c.customer_id, c.annual_income
    )
    
    SELECT 
        lf.is_default,
        lf.loan_id,
        
        -- Demographic
        df.age, df.age_group, df.gender, df.marital_status,
        df.dependents, df.education, df.employment_type,
        df.annual_income, df.log_annual_income,
        df.employment_stability_ratio, df.income_per_dependent,
        df.city_tier, df.verification_score,
        
        -- Bureau
        bf.credit_score, bf.credit_score_tier,
        bf.total_accounts, bf.active_accounts, bf.active_account_ratio,
        bf.overdue_30_count, bf.overdue_60_count, bf.overdue_90_count,
        bf.delinquency_severity_score, bf.has_serious_delinquency,
        bf.credit_utilization, bf.utilization_bucket,
        bf.enquiries_last_6m, bf.enquiry_velocity,
        bf.credit_history_length_months, bf.is_thin_file,
        
        -- Loan
        lf.loan_amount, lf.log_loan_amount, lf.loan_term_months, lf.term_bucket,
        lf.interest_rate, lf.rate_tier, lf.emi_amount,
        lf.loan_type, lf.is_secured, lf.dti_ratio,
        lf.emi_to_income_ratio, lf.loan_to_income_ratio,
        lf.interest_to_principal_ratio,
        
        -- Customer aggregate (no leaky features)
        ca.total_loans,
        ca.total_exposure_to_income, ca.loan_type_diversity
        
    FROM loan_features lf
    LEFT JOIN demographic_features df ON lf.customer_id = df.customer_id
    LEFT JOIN bureau_features bf ON lf.customer_id = bf.customer_id
    LEFT JOIN customer_agg ca ON lf.customer_id = ca.customer_id;
    """
    
    print("  Executing SQL feature engineering queries...")
    df = pd.read_sql_query(query, conn)
    print(f"    → Generated feature table: {df.shape[0]} rows × {df.shape[1]} columns")
    
    return df


def main():
    print("="*60)
    print("  CREDIT RISK — SQL FEATURE ENGINEERING")
    print("="*60)
    
    conn = connect_db()
    
    # Build ML feature table
    features_df = build_feature_table(conn)
    
    # Save
    features_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n  ✓ ML features saved to: {OUTPUT_PATH}")
    
    # Quick stats
    print(f"\n  Feature Summary:")
    print(f"  {'─'*40}")
    print(f"  Total features:    {features_df.shape[1] - 2}")  # exclude target + loan_id
    print(f"  Total samples:     {features_df.shape[0]:,}")
    print(f"  Default rate:      {features_df['is_default'].mean():.1%}")
    print(f"  Missing values:    {features_df.isna().sum().sum():,} ({features_df.isna().sum().sum() / features_df.size * 100:.1f}%)")
    print(f"\n  Top 5 features with missing values:")
    missing = features_df.isna().sum().sort_values(ascending=False).head()
    for col, count in missing.items():
        if count > 0:
            print(f"    {col}: {count} ({count/len(features_df)*100:.1f}%)")
    
    conn.close()
    print(f"\n  ✓ Done!")


if __name__ == '__main__':
    main()
