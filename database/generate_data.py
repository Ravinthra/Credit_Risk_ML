"""
Credit Risk ML — Synthetic Data Generator
==========================================
Generates realistic loan default data WITHOUT needing PostgreSQL.
Uses SQLite as a lightweight alternative, then exports to CSV for modeling.

WHY SYNTHETIC DATA?
───────────────────
1. Real credit data is heavily regulated (PII, GDPR, RBI guidelines)
2. Production datasets are proprietary — can't share on GitHub
3. Synthetic data lets us control class balance and feature distributions
4. We can inject realistic correlations (e.g., low income → higher default)

DESIGN PHILOSOPHY:
──────────────────
- Distributions mimic real Indian fintech data
- Default rate ~15-20% (realistic for unsecured personal loans)
- Correlated features (income ↔ credit score, DPD ↔ default)
- Missing values injected realistically (not random — MNAR pattern)
"""

import numpy as np
import pandas as pd
import sqlite3
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# ─── Configuration ───────────────────────────────────────────
RANDOM_SEED = 42
N_CUSTOMERS = 5000
N_LOANS = 8000          # Some customers have multiple loans
DEFAULT_RATE = 0.18     # ~18% default rate (realistic for personal loans)

np.random.seed(RANDOM_SEED)

# ─── Output Paths ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "credit_risk.db"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def generate_customers(n: int) -> pd.DataFrame:
    """
    Generate customer demographic data.
    
    WHY THESE DISTRIBUTIONS?
    - Age: Beta distribution shifted to 22-65 range (working population)
    - Income: Log-normal (realistic wealth distribution — heavily right-skewed)
    - Employment: Weighted to reflect Indian workforce composition
    """
    print(f"  Generating {n} customers...")
    
    # Age: Beta distribution (peaks around 30-40)
    age = np.clip(
        (np.random.beta(2, 3, n) * 50 + 22).astype(int),
        18, 75
    )
    
    gender = np.random.choice(
        ['Male', 'Female', 'Other'], n,
        p=[0.62, 0.36, 0.02]  # Reflects Indian lending demographics
    )
    
    marital_status = np.random.choice(
        ['Single', 'Married', 'Divorced', 'Widowed'], n,
        p=[0.30, 0.55, 0.10, 0.05]
    )
    
    # Dependents: Correlated with marital status
    dependents = np.where(
        np.isin(marital_status, ['Married']),
        np.random.choice([0, 1, 2, 3, 4], n, p=[0.15, 0.25, 0.35, 0.15, 0.10]),
        np.random.choice([0, 1, 2], n, p=[0.70, 0.20, 0.10])
    )
    
    education = np.random.choice(
        ['High School', 'Bachelor', 'Master', 'PhD', 'Other'], n,
        p=[0.25, 0.40, 0.20, 0.05, 0.10]
    )
    
    employment_type = np.random.choice(
        ['Salaried', 'Self-Employed', 'Business', 'Freelance', 'Unemployed'], n,
        p=[0.45, 0.20, 0.15, 0.12, 0.08]
    )
    
    # Annual Income: Log-normal, correlated with education + employment
    base_income = np.random.lognormal(mean=12.5, sigma=0.8, size=n)
    
    # Education multiplier
    edu_mult = np.where(education == 'PhD', 1.8,
               np.where(education == 'Master', 1.4,
               np.where(education == 'Bachelor', 1.1,
               np.where(education == 'High School', 0.75, 0.9))))
    
    # Employment multiplier
    emp_mult = np.where(employment_type == 'Business', 1.5,
               np.where(employment_type == 'Salaried', 1.1,
               np.where(employment_type == 'Self-Employed', 1.0,
               np.where(employment_type == 'Freelance', 0.85, 0.3))))
    
    annual_income = np.clip(base_income * edu_mult * emp_mult, 100000, 10000000)
    annual_income = np.round(annual_income, -3)  # Round to nearest thousand
    
    # Years employed: Correlated with age
    max_years = np.clip(age - 20, 0, 40)
    years_employed = np.round(np.random.uniform(0, max_years) * 0.6, 1)
    years_employed = np.where(employment_type == 'Unemployed', 0, years_employed)
    
    city_tier = np.random.choice([1, 2, 3], n, p=[0.40, 0.35, 0.25])
    
    state = np.random.choice(
        ['Maharashtra', 'Karnataka', 'Tamil Nadu', 'Delhi', 'Telangana',
         'Gujarat', 'Rajasthan', 'UP', 'West Bengal', 'Kerala',
         'MP', 'Bihar', 'Haryana', 'Punjab', 'Odisha'],
        n,
        p=[0.15, 0.12, 0.10, 0.10, 0.08,
           0.08, 0.07, 0.07, 0.05, 0.05,
           0.04, 0.03, 0.03, 0.02, 0.01]
    )
    
    phone_verified = np.random.choice([True, False], n, p=[0.85, 0.15])
    email_verified = np.random.choice([True, False], n, p=[0.70, 0.30])
    
    df = pd.DataFrame({
        'customer_id': range(1, n + 1),
        'age': age,
        'gender': gender,
        'marital_status': marital_status,
        'dependents': dependents,
        'education': education,
        'employment_type': employment_type,
        'years_employed': years_employed,
        'annual_income': annual_income,
        'city_tier': city_tier,
        'state': state,
        'phone_verified': phone_verified,
        'email_verified': email_verified
    })
    
    return df


def generate_credit_history(customers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate credit bureau data correlated with customer attributes.
    
    WHY CORRELATIONS MATTER:
    - High income → higher credit score (on average)
    - More accounts → longer credit history
    - Lower score → more delinquencies (circular but realistic)
    """
    n = len(customers_df)
    print(f"  Generating credit history for {n} customers...")
    
    income = customers_df['annual_income'].values
    age = customers_df['age'].values
    
    # Credit Score: Correlated with income (0.4 correlation) and age (0.2)
    income_percentile = np.argsort(np.argsort(income)) / n
    age_percentile = np.argsort(np.argsort(age)) / n
    
    base_score = 300 + (income_percentile * 0.4 + age_percentile * 0.2 + 
                        np.random.uniform(0, 0.4, n)) * 600
    credit_score = np.clip(base_score.astype(int), 300, 900)
    
    # Account counts: Correlated with age
    total_accounts = np.clip(
        (age / 10 + np.random.poisson(2, n)).astype(int), 0, 30
    )
    active_ratio = np.random.uniform(0.2, 0.8, n)
    active_accounts = np.clip((total_accounts * active_ratio).astype(int), 0, total_accounts)
    closed_accounts = total_accounts - active_accounts
    
    # Delinquency: Inversely correlated with credit score
    score_norm = (credit_score - 300) / 600  # 0 to 1
    delinquency_risk = 1 - score_norm  # Higher risk for lower scores
    
    overdue_30 = np.random.poisson(delinquency_risk * 3, n).astype(int)
    overdue_60 = np.random.poisson(delinquency_risk * 1.5, n).astype(int)
    overdue_90 = np.random.poisson(delinquency_risk * 0.8, n).astype(int)
    
    # Credit utilization: Higher for lower scores
    utilization_base = np.clip(0.8 - score_norm * 0.6 + np.random.normal(0, 0.15, n), 0, 1)
    total_credit_limit = np.clip(income * np.random.uniform(0.5, 3, n), 50000, 5000000)
    current_balance = total_credit_limit * utilization_base
    
    # Enquiries: More for risky profiles
    enquiries_6m = np.random.poisson(delinquency_risk * 2 + 0.5, n).astype(int)
    enquiries_12m = enquiries_6m + np.random.poisson(1, n).astype(int)
    
    # Credit history length: Correlated with age
    credit_history_months = np.clip(
        ((age - 18) * np.random.uniform(0.3, 0.9, n) * 12).astype(int),
        0, 480
    )
    
    df = pd.DataFrame({
        'credit_id': range(1, n + 1),
        'customer_id': customers_df['customer_id'].values,
        'credit_score': credit_score,
        'total_accounts': total_accounts,
        'active_accounts': active_accounts,
        'closed_accounts': closed_accounts,
        'overdue_30_count': overdue_30,
        'overdue_60_count': overdue_60,
        'overdue_90_count': overdue_90,
        'total_credit_limit': np.round(total_credit_limit, 2),
        'current_balance': np.round(current_balance, 2),
        'enquiries_last_6m': enquiries_6m,
        'enquiries_last_12m': enquiries_12m,
        'credit_history_length_months': credit_history_months
    })
    
    return df


def generate_loan_applications(customers_df: pd.DataFrame, 
                                credit_df: pd.DataFrame,
                                n_loans: int) -> pd.DataFrame:
    """
    Generate loan applications with realistic default correlation.
    
    THE DEFAULT LOGIC:
    ──────────────────
    Default probability is modeled as a logistic function of:
    - Credit score (negative: higher score → lower default)
    - DTI ratio (positive: higher DTI → higher default)
    - Employment stability (negative: stable → lower default)
    - Loan type risk (unsecured > secured)
    
    This mimics how actual credit scoring models work internally.
    """
    print(f"  Generating {n_loans} loan applications...")
    
    # Assign customers to loans (some customers get multiple loans)
    # WHY WEIGHTED? Riskier customers tend to apply for more loans
    customer_ids = np.random.choice(
        customers_df['customer_id'].values,
        size=n_loans,
        replace=True
    )
    
    # Merge customer and credit data for correlation
    customer_lookup = customers_df.set_index('customer_id')
    credit_lookup = credit_df.set_index('customer_id')
    
    incomes = customer_lookup.loc[customer_ids, 'annual_income'].values
    monthly_incomes = incomes / 12
    ages = customer_lookup.loc[customer_ids, 'age'].values
    emp_types = customer_lookup.loc[customer_ids, 'employment_type'].values
    years_emp = customer_lookup.loc[customer_ids, 'years_employed'].values
    credit_scores = credit_lookup.loc[customer_ids, 'credit_score'].values
    
    # Loan type distribution
    loan_types = np.random.choice(
        ['Personal', 'Home', 'Auto', 'Education', 'Business', 'Credit Card', 'Gold'],
        n_loans,
        p=[0.30, 0.15, 0.15, 0.10, 0.15, 0.10, 0.05]
    )
    
    # Loan amount: Depends on loan type and income
    amount_multiplier = np.where(loan_types == 'Home', np.random.uniform(3, 8, n_loans),
                        np.where(loan_types == 'Auto', np.random.uniform(0.5, 2, n_loans),
                        np.where(loan_types == 'Personal', np.random.uniform(0.2, 1.5, n_loans),
                        np.where(loan_types == 'Education', np.random.uniform(0.5, 3, n_loans),
                        np.where(loan_types == 'Business', np.random.uniform(1, 5, n_loans),
                        np.where(loan_types == 'Credit Card', np.random.uniform(0.05, 0.3, n_loans),
                        np.random.uniform(0.1, 0.5, n_loans)))))))
    
    loan_amount = np.clip(incomes * amount_multiplier, 10000, 50000000)
    loan_amount = np.round(loan_amount, -3)
    
    # Loan term (months)
    term_map = {
        'Home': lambda: np.random.choice([120, 180, 240, 300, 360]),
        'Auto': lambda: np.random.choice([36, 48, 60, 72, 84]),
        'Personal': lambda: np.random.choice([12, 24, 36, 48, 60]),
        'Education': lambda: np.random.choice([36, 48, 60, 84, 120]),
        'Business': lambda: np.random.choice([12, 24, 36, 60]),
        'Credit Card': lambda: np.random.choice([3, 6, 12, 24]),
        'Gold': lambda: np.random.choice([6, 12, 24])
    }
    loan_term = np.array([term_map[lt]() for lt in loan_types])
    
    # Interest rate: Higher for riskier profiles + unsecured loans
    base_rate = np.where(loan_types == 'Home', np.random.uniform(7, 10, n_loans),
                np.where(loan_types == 'Auto', np.random.uniform(8, 12, n_loans),
                np.where(loan_types == 'Gold', np.random.uniform(7, 11, n_loans),
                np.where(loan_types == 'Education', np.random.uniform(8, 13, n_loans),
                np.random.uniform(10, 24, n_loans)))))
    
    # Credit score adjustment: Lower score → higher rate
    score_adj = (700 - credit_scores) / 100  # Positive for low scores
    interest_rate = np.clip(np.round(base_rate + score_adj, 2), 5, 36)
    
    # EMI calculation: Standard formula
    monthly_rate = interest_rate / (12 * 100)
    emi = loan_amount * monthly_rate * (1 + monthly_rate)**loan_term / ((1 + monthly_rate)**loan_term - 1)
    emi = np.round(emi, 2)
    
    # Is secured?
    is_secured = np.isin(loan_types, ['Home', 'Auto', 'Gold'])
    collateral_value = np.where(
        is_secured,
        loan_amount * np.random.uniform(1.0, 1.5, n_loans),
        0
    )
    
    # DTI ratio
    dti_ratio = np.where(monthly_incomes > 0, emi / monthly_incomes, 0)
    dti_ratio = np.clip(dti_ratio, 0, 1)
    
    # LTV ratio
    ltv_ratio = np.where(
        (is_secured) & (collateral_value > 0),
        loan_amount / collateral_value,
        0
    )
    
    # ─── DEFAULT PREDICTION (ground truth generation) ───
    # This is the "god model" that determines our labels.
    # In reality, we'd observe defaults over time.
    
    score_norm = (credit_scores - 300) / 600  # 0 to 1 (higher = better)
    
    # Default probability components
    p_score = (1 - score_norm) * 0.35       # Low score → high default
    p_dti = np.clip(dti_ratio, 0, 1) * 0.20  # High DTI → high default
    p_emp = np.where(emp_types == 'Unemployed', 0.15,
            np.where(emp_types == 'Freelance', 0.08,
            np.where(years_emp < 1, 0.06, 0.02)))
    p_type = np.where(loan_types == 'Personal', 0.08,
             np.where(loan_types == 'Credit Card', 0.07,
             np.where(loan_types == 'Business', 0.06,
             np.where(loan_types == 'Home', 0.02, 0.04))))
    
    # Add noise
    p_noise = np.random.uniform(-0.05, 0.10, n_loans)
    
    default_prob = np.clip(p_score + p_dti + p_emp + p_type + p_noise, 0.02, 0.85)
    is_default = np.random.binomial(1, default_prob).astype(bool)
    
    # Risk Grade (based on default probability)
    risk_grade = np.where(default_prob < 0.10, 'A',
                 np.where(default_prob < 0.20, 'B',
                 np.where(default_prob < 0.35, 'C',
                 np.where(default_prob < 0.50, 'D',
                 np.where(default_prob < 0.65, 'E', 'F')))))
    
    # Loan purpose
    purposes = np.random.choice(
        ['Home Renovation', 'Medical Emergency', 'Wedding', 'Debt Consolidation',
         'Business Expansion', 'Vehicle Purchase', 'Education Fees', 'Travel',
         'Electronics', 'Working Capital', 'Inventory Finance', 'New Home',
         'Home Improvement', 'Personal Need'],
        n_loans
    )
    
    # Status
    status = np.where(is_default, 
                      np.random.choice(['Written-Off', 'Active'], n_loans, p=[0.3, 0.7]),
                      np.random.choice(['Active', 'Closed', 'Disbursed'], n_loans, p=[0.4, 0.4, 0.2]))
    
    # Dates
    app_dates = pd.date_range(end='2025-12-31', periods=n_loans, freq=None)
    app_dates = pd.to_datetime(
        np.random.choice(pd.date_range('2020-01-01', '2025-06-30'), n_loans)
    )
    
    df = pd.DataFrame({
        'loan_id': range(1, n_loans + 1),
        'customer_id': customer_ids,
        'loan_amount': loan_amount,
        'loan_term_months': loan_term,
        'interest_rate': interest_rate,
        'emi_amount': emi,
        'loan_type': loan_types,
        'loan_purpose': purposes,
        'is_secured': is_secured,
        'collateral_value': np.round(collateral_value, 2),
        'dti_ratio': np.round(dti_ratio, 4),
        'ltv_ratio': np.round(ltv_ratio, 4),
        'application_date': app_dates,
        'status': status,
        'is_default': is_default,
        'risk_grade': risk_grade
    })
    
    return df


def generate_repayment_history(loans_df: pd.DataFrame,
                                customers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate monthly repayment records for each loan.
    
    WHY THIS IS COMPLEX:
    - Defaulters show gradually worsening payment behavior
    - Non-defaulters are mostly on-time with occasional lapses
    - Payment amounts include partial payments for stressed borrowers
    """
    print(f"  Generating repayment history (this takes a moment)...")
    
    records = []
    customer_lookup = customers_df.set_index('customer_id')
    
    for _, loan in loans_df.iterrows():
        # Generate installments per loan (not full term)
        max_possible = int(loan['loan_term_months'])
        if max_possible <= 6:
            n_installments = max_possible
        else:
            n_installments = min(max_possible, np.random.randint(6, min(25, max_possible + 1)))
        
        is_default = loan['is_default']
        emi = loan['emi_amount']
        
        # Base late payment probability
        base_late_prob = 0.35 if is_default else 0.08
        
        for i in range(1, n_installments + 1):
            due_date = pd.Timestamp(loan['application_date']) + pd.DateOffset(months=i)
            
            # Defaulters: late probability INCREASES over time (deterioration)
            # Non-defaulters: stable low probability
            if is_default:
                late_prob = min(base_late_prob + (i / n_installments) * 0.4, 0.85)
            else:
                late_prob = base_late_prob + np.random.uniform(-0.02, 0.02)
            
            is_late = np.random.random() < late_prob
            
            if is_late:
                days_past_due = int(np.random.exponential(30) + 5)
                if is_default and i > n_installments * 0.6:
                    days_past_due = int(days_past_due * 2)  # Worse DPD for late-stage defaulters
                days_past_due = min(days_past_due, 365)
            else:
                days_past_due = 0
            
            # Payment amount
            if days_past_due > 90:
                # Severely delinquent: likely partial or no payment
                amount_paid = round(emi * np.random.uniform(0, 0.5), 2)
            elif is_late:
                amount_paid = round(emi * np.random.uniform(0.7, 1.0), 2)
            else:
                amount_paid = round(emi * np.random.uniform(0.98, 1.02), 2)  # Minor variations
            
            is_partial = amount_paid < emi * 0.95
            
            payment_date = due_date + pd.Timedelta(days=days_past_due) if amount_paid > 0 else None
            
            payment_method = np.random.choice(
                ['Auto-Debit', 'UPI', 'NEFT', 'Cash', 'Cheque', 'Other'],
                p=[0.35, 0.25, 0.15, 0.10, 0.10, 0.05]
            )
            
            # Outstanding balance (simplified)
            remaining_installments = loan['loan_term_months'] - i
            outstanding = round(emi * remaining_installments, 2)
            
            records.append({
                'loan_id': loan['loan_id'],
                'customer_id': loan['customer_id'],
                'installment_number': i,
                'due_date': due_date.strftime('%Y-%m-%d'),
                'payment_date': payment_date.strftime('%Y-%m-%d') if payment_date else None,
                'amount_due': round(emi, 2),
                'amount_paid': amount_paid,
                'days_past_due': days_past_due,
                'is_late': is_late,
                'is_partial': is_partial,
                'outstanding_balance': max(outstanding, 0),
                'payment_method': payment_method
            })
    
    df = pd.DataFrame(records)
    df.insert(0, 'repayment_id', range(1, len(df) + 1))
    print(f"    → Generated {len(df)} repayment records")
    return df


def inject_missing_values(customers_df: pd.DataFrame, 
                          credit_df: pd.DataFrame,
                          loans_df: pd.DataFrame) -> tuple:
    """
    Inject realistic missing values (MNAR — Missing Not At Random).
    
    WHY MNAR?
    ─────────
    In real credit data, missingness is NOT random:
    - Self-employed people often don't report income accurately → missing income
    - New-to-credit customers → missing credit history
    - Unverified customers → missing contact info
    
    This is critical for interviews: "How did you handle missing values?"
    → "Our data had MNAR patterns — e.g., credit history was missing for
       thin-file customers, not randomly. So I used indicator variables
       alongside imputation to capture the informativeness of missingness."
    """
    print("  Injecting realistic missing values (MNAR pattern)...")
    
    customers = customers_df.copy()
    credit = credit_df.copy()
    loans = loans_df.copy()
    
    # Customers: ~5% missing years_employed (mostly self-employed/freelance)
    mask = (customers['employment_type'].isin(['Self-Employed', 'Freelance'])) & \
           (np.random.random(len(customers)) < 0.15)
    customers.loc[mask, 'years_employed'] = np.nan
    
    # Customers: ~3% missing education
    customers.loc[np.random.random(len(customers)) < 0.03, 'education'] = np.nan
    
    # Credit: ~8% missing credit_score (thin file / new to credit)
    thin_file_mask = credit['credit_history_length_months'] < 6
    credit.loc[thin_file_mask & (np.random.random(len(credit)) < 0.5), 'credit_score'] = np.nan
    credit.loc[np.random.random(len(credit)) < 0.03, 'credit_score'] = np.nan
    
    # Credit: ~5% missing enquiry data
    credit.loc[np.random.random(len(credit)) < 0.05, 'enquiries_last_6m'] = np.nan
    credit.loc[np.random.random(len(credit)) < 0.05, 'enquiries_last_12m'] = np.nan
    
    # Loans: ~4% missing collateral value (for secured loans — data entry errors)
    secured_mask = loans['is_secured'] == True
    loans.loc[secured_mask & (np.random.random(len(loans)) < 0.08), 'collateral_value'] = np.nan
    
    # Loans: ~6% missing interest rate (pre-approval applications)
    loans.loc[np.random.random(len(loans)) < 0.02, 'interest_rate'] = np.nan
    
    n_missing = customers.isna().sum().sum() + credit.isna().sum().sum() + loans.isna().sum().sum()
    print(f"    → Injected {n_missing} missing values across all tables")
    
    return customers, credit, loans


def save_to_sqlite(customers, credit, loans, repayments, db_path):
    """Save all tables to SQLite database."""
    print(f"  Saving to SQLite: {db_path}")
    
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    
    customers.to_sql('customers', conn, index=False, if_exists='replace')
    credit.to_sql('credit_history', conn, index=False, if_exists='replace')
    loans.to_sql('loan_applications', conn, index=False, if_exists='replace')
    repayments.to_sql('repayment_history', conn, index=False, if_exists='replace')
    
    conn.close()
    print(f"    → Database saved successfully")


def save_to_csv(customers, credit, loans, repayments, data_dir):
    """Export all tables as CSV for easy access."""
    print(f"  Exporting CSVs to: {data_dir}")
    
    customers.to_csv(data_dir / 'customers.csv', index=False)
    credit.to_csv(data_dir / 'credit_history.csv', index=False)
    loans.to_csv(data_dir / 'loan_applications.csv', index=False)
    repayments.to_csv(data_dir / 'repayment_history.csv', index=False)
    
    print(f"    → All CSVs exported")


def print_summary(customers, credit, loans, repayments):
    """Print dataset summary statistics."""
    print("\n" + "="*60)
    print("  DATASET SUMMARY")
    print("="*60)
    print(f"  Customers:     {len(customers):>8,} rows")
    print(f"  Credit History: {len(credit):>7,} rows")
    print(f"  Loan Applications: {len(loans):>4,} rows")
    print(f"  Repayment Records: {len(repayments):>4,} rows")
    print(f"\n  Default Rate:  {loans['is_default'].mean():.1%}")
    print(f"  Defaults:      {loans['is_default'].sum():>8,}")
    print(f"  Non-Defaults:  {(~loans['is_default']).sum():>8,}")
    print(f"\n  Missing Values:")
    for name, df in [('Customers', customers), ('Credit', credit), 
                     ('Loans', loans), ('Repayments', repayments)]:
        n_miss = df.isna().sum().sum()
        pct = n_miss / (df.shape[0] * df.shape[1]) * 100
        print(f"    {name}: {n_miss:,} ({pct:.1f}%)")
    print("="*60)


def main():
    print("="*60)
    print("  CREDIT RISK — SYNTHETIC DATA GENERATOR")
    print("="*60)
    print(f"  Seed: {RANDOM_SEED}")
    print(f"  Target: {N_CUSTOMERS} customers, {N_LOANS} loans")
    print(f"  Default rate: ~{DEFAULT_RATE:.0%}")
    print("-"*60)
    
    # Step 1: Generate base tables
    customers = generate_customers(N_CUSTOMERS)
    credit = generate_credit_history(customers)
    loans = generate_loan_applications(customers, credit, N_LOANS)
    repayments = generate_repayment_history(loans, customers)
    
    # Step 2: Inject missing values (MNAR)
    customers, credit, loans = inject_missing_values(customers, credit, loans)
    
    # Step 3: Save
    save_to_sqlite(customers, credit, loans, repayments, DB_PATH)
    save_to_csv(customers, credit, loans, repayments, DATA_DIR)
    
    # Step 4: Summary
    print_summary(customers, credit, loans, repayments)
    
    print(f"\n  ✓ Data generation complete!")
    print(f"  ✓ Database: {DB_PATH}")
    print(f"  ✓ CSVs:     {DATA_DIR}")


if __name__ == '__main__':
    main()
