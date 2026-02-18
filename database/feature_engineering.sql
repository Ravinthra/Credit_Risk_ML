-- ============================================================
-- CREDIT RISK — SQL FEATURE ENGINEERING
-- ============================================================
--
-- WHY SQL FOR FEATURE ENGINEERING?
-- ─────────────────────────────────
-- 1. Performance: SQL engines are optimized for aggregations on
--    millions of rows. Doing this in pandas is 10-100x slower.
-- 2. Production-ready: In real ML pipelines, features are computed
--    in SQL (Spark SQL / BigQuery / Redshift) not in notebooks.
-- 3. Reproducibility: SQL queries are declarative and deterministic.
-- 4. Interview Signal: Shows you understand data engineering, not
--    just model.fit().
--
-- Feature Categories:
--   A. Demographic Features (from customers table)
--   B. Credit Bureau Features (from credit_history)
--   C. Loan-Level Features (from loan_applications)
--   D. Behavioral / Repayment Features (from repayment_history)
--   E. Interaction / Derived Features (cross-table)
-- ============================================================


-- ──────────────────────────────────────────
-- A. DEMOGRAPHIC FEATURES
-- ──────────────────────────────────────────
-- WHY? Demographics alone are weak predictors, but they add
-- incremental lift when combined with bureau + behavioral data.
-- Also required for regulatory fair lending analysis.

-- A1: Age bucketing (non-linear relationship with default)
-- WHY BUCKETS? Age vs default is NOT linear. Young borrowers (18-25)
-- and very old borrowers (65+) have higher default rates than 35-50.
-- Bucketing captures this non-linearity for linear models.

CREATE OR REPLACE VIEW v_demographic_features AS
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
    
    -- Income features
    annual_income,
    monthly_income,
    
    -- WHY LOG INCOME? Income is heavily right-skewed.
    -- Log transform makes it more normally distributed,
    -- which helps linear models significantly.
    LN(GREATEST(annual_income, 1)) AS log_annual_income,
    
    -- Income stability proxy
    -- WHY? Years employed relative to age indicates career stability.
    -- Someone 40 years old with 10 years at same company = stable.
    CASE 
        WHEN age > 0 THEN years_employed / age 
        ELSE 0 
    END AS employment_stability_ratio,
    
    -- Income per dependent
    -- WHY? A ₹10L salary with 0 dependents vs 5 dependents = very different risk
    CASE 
        WHEN dependents > 0 THEN annual_income / dependents
        ELSE annual_income  -- no dependents = all income is available
    END AS income_per_dependent,
    
    city_tier,
    
    -- Verification flags (proxy for data quality / fraud risk)
    phone_verified,
    email_verified,
    CASE 
        WHEN phone_verified AND email_verified THEN 2
        WHEN phone_verified OR email_verified THEN 1
        ELSE 0
    END AS verification_score

FROM customers;


-- ──────────────────────────────────────────
-- B. CREDIT BUREAU FEATURES
-- ──────────────────────────────────────────
-- WHY? Bureau features are the highest-signal predictors.
-- A model using ONLY bureau data can achieve 70-80% of the
-- full model's performance. This is well-known in industry.

CREATE OR REPLACE VIEW v_bureau_features AS
SELECT 
    customer_id,
    credit_score,
    
    -- Credit score buckets (for interpretability + non-linear capture)
    -- WHY? Score ranges map to risk tiers used by underwriters.
    CASE 
        WHEN credit_score >= 750 THEN 'Excellent'
        WHEN credit_score >= 700 THEN 'Good'
        WHEN credit_score >= 650 THEN 'Fair'
        WHEN credit_score >= 550 THEN 'Poor'
        ELSE 'Very_Poor'
    END AS credit_score_tier,
    
    -- Account-level features
    total_accounts,
    active_accounts,
    closed_accounts,
    
    -- Account activity ratio
    -- WHY? Someone with 10 accounts, 8 active = heavily leveraged
    -- vs 10 accounts, 2 active = reduced exposure
    CASE 
        WHEN total_accounts > 0 THEN active_accounts::NUMERIC / total_accounts
        ELSE 0 
    END AS active_account_ratio,
    
    -- Delinquency features (THE most important sub-group)
    overdue_30_count,
    overdue_60_count,
    overdue_90_count,
    
    -- Total delinquency severity score
    -- WHY WEIGHTED? A 90-DPD is MUCH worse than a 30-DPD.
    -- Weights reflect the exponential increase in risk.
    (overdue_30_count * 1 + overdue_60_count * 3 + overdue_90_count * 10) 
        AS delinquency_severity_score,
    
    -- Has any serious delinquency? (binary flag)
    -- WHY? Even ONE 90+ DPD is a strong signal. Binary flags
    -- capture the "threshold effect" that continuous features miss.
    CASE WHEN overdue_90_count > 0 THEN 1 ELSE 0 END AS has_serious_delinquency,
    
    -- Credit utilization (already computed in schema, but we re-derive for clarity)
    credit_utilization,
    
    -- Utilization buckets
    -- WHY? Utilization risk is non-linear:
    -- 0-30% = healthy, 30-60% = moderate, 60-80% = high, 80%+ = critical
    CASE 
        WHEN credit_utilization <= 0.30 THEN 'Low'
        WHEN credit_utilization <= 0.60 THEN 'Moderate'
        WHEN credit_utilization <= 0.80 THEN 'High'
        ELSE 'Critical'
    END AS utilization_bucket,
    
    -- Enquiry features
    enquiries_last_6m,
    enquiries_last_12m,
    
    -- Enquiry velocity (recent vs total)
    -- WHY? 4 enquiries in 6 months when you had 5 in 12 months = sudden spike
    -- This "acceleration" signal catches desperation better than raw count.
    CASE 
        WHEN enquiries_last_12m > 0 
        THEN enquiries_last_6m::NUMERIC / enquiries_last_12m
        ELSE 0 
    END AS enquiry_velocity,
    
    -- Credit history length
    credit_history_length_months,
    
    -- Thin file indicator
    -- WHY? New-to-credit customers have <12 months history.
    -- They need different scoring approaches (alternative data, etc.)
    CASE 
        WHEN credit_history_length_months < 12 THEN 1 
        ELSE 0 
    END AS is_thin_file

FROM credit_history;


-- ──────────────────────────────────────────
-- C. LOAN-LEVEL FEATURES
-- ──────────────────────────────────────────
-- WHY? The characteristics of the loan itself affect risk.
-- A ₹50L personal loan at 18% is riskier than a ₹50L home loan at 8%.

CREATE OR REPLACE VIEW v_loan_features AS
SELECT 
    la.loan_id,
    la.customer_id,
    la.loan_amount,
    la.loan_term_months,
    la.interest_rate,
    la.emi_amount,
    la.loan_type,
    la.is_secured,
    la.collateral_value,
    la.dti_ratio,
    la.ltv_ratio,
    la.is_default,
    
    -- Log loan amount (right-skewed, like income)
    LN(GREATEST(la.loan_amount, 1)) AS log_loan_amount,
    
    -- Loan term buckets
    -- WHY? Short-term loans (12m) have different risk than long-term (240m)
    -- Short-term: higher EMI stress. Long-term: more time for life events.
    CASE 
        WHEN la.loan_term_months <= 12 THEN 'Short'
        WHEN la.loan_term_months <= 60 THEN 'Medium'
        WHEN la.loan_term_months <= 120 THEN 'Long'
        ELSE 'Very_Long'
    END AS term_bucket,
    
    -- Interest rate tier
    -- WHY? High interest rates are assigned to risky borrowers (adverse selection).
    -- Rate itself becomes a feature that encodes the lender's initial risk assessment.
    CASE 
        WHEN la.interest_rate <= 10 THEN 'Prime'
        WHEN la.interest_rate <= 15 THEN 'Near_Prime'
        WHEN la.interest_rate <= 20 THEN 'Subprime'
        ELSE 'Deep_Subprime'
    END AS rate_tier,
    
    -- Collateral coverage ratio (for secured loans)
    -- WHY? If collateral > loan, the lender has a safety net.
    -- LTV > 1.0 = underwater, LTV < 0.5 = well-secured
    CASE 
        WHEN la.is_secured AND la.collateral_value > 0 
        THEN la.loan_amount / la.collateral_value
        ELSE NULL  -- not applicable for unsecured
    END AS effective_ltv,
    
    -- EMI-to-income ratio (THE most important affordability metric)
    -- WHY? Banks use FOIR (Fixed Obligation to Income Ratio).
    -- Industry rule: FOIR > 50% = high risk, > 60% = likely rejection.
    CASE 
        WHEN c.monthly_income > 0 
        THEN la.emi_amount / c.monthly_income
        ELSE NULL 
    END AS emi_to_income_ratio,
    
    -- Loan amount to annual income ratio
    -- WHY? Captures leverage. Borrowing 10x your income = extreme risk.
    CASE 
        WHEN c.annual_income > 0 
        THEN la.loan_amount / c.annual_income
        ELSE NULL 
    END AS loan_to_income_ratio,
    
    -- Total interest payable over loan life
    -- WHY? High total interest = long-term burden. Useful interaction feature.
    (la.emi_amount * la.loan_term_months) - la.loan_amount AS total_interest_payable,
    
    -- Interest as percentage of principal
    CASE 
        WHEN la.loan_amount > 0 
        THEN ((la.emi_amount * la.loan_term_months) - la.loan_amount) / la.loan_amount
        ELSE 0 
    END AS interest_to_principal_ratio

FROM loan_applications la
JOIN customers c ON la.customer_id = c.customer_id;


-- ──────────────────────────────────────────
-- D. BEHAVIORAL / REPAYMENT FEATURES
-- ──────────────────────────────────────────
-- WHY? Behavioral data is the GOLD MINE of credit risk modeling.
-- It captures HOW people pay, not just what static data says.
-- This is what separates a ₹15L ML role from a ₹8L analyst role.
--
-- Interview Insight: "Tell me about feature engineering you've done"
-- → Discuss these behavioral aggregations. They show domain depth.

CREATE OR REPLACE VIEW v_repayment_features AS
SELECT 
    loan_id,
    customer_id,
    
    -- ─── Payment Counting Features ───
    COUNT(*) AS total_installments_due,
    COUNT(CASE WHEN amount_paid > 0 THEN 1 END) AS installments_paid,
    COUNT(CASE WHEN is_late THEN 1 END) AS late_payment_count,
    COUNT(CASE WHEN is_partial THEN 1 END) AS partial_payment_count,
    COUNT(CASE WHEN amount_paid = 0 AND payment_date IS NULL THEN 1 END) AS missed_payment_count,
    
    -- ─── Payment Ratios ───
    -- WHY RATIOS? Absolute counts are biased by loan age.
    -- A 2-year loan with 3 late payments ≠ a 10-year loan with 3 late payments.
    ROUND(
        COUNT(CASE WHEN is_late THEN 1 END)::NUMERIC / NULLIF(COUNT(*), 0), 4
    ) AS late_payment_ratio,
    
    ROUND(
        COUNT(CASE WHEN is_partial THEN 1 END)::NUMERIC / NULLIF(COUNT(*), 0), 4
    ) AS partial_payment_ratio,
    
    -- ─── DPD (Days Past Due) Statistics ───
    -- WHY MULTIPLE AGGREGATIONS? Mean DPD captures average behavior,
    -- Max DPD captures worst episodes, Std captures consistency.
    ROUND(AVG(days_past_due)::NUMERIC, 2) AS avg_dpd,
    MAX(days_past_due) AS max_dpd,
    ROUND(STDDEV(days_past_due)::NUMERIC, 2) AS std_dpd,
    
    -- Ever 90+ DPD? (NPA indicator)
    MAX(CASE WHEN days_past_due >= 90 THEN 1 ELSE 0 END) AS ever_90_plus_dpd,
    -- Ever 60+ DPD?
    MAX(CASE WHEN days_past_due >= 60 THEN 1 ELSE 0 END) AS ever_60_plus_dpd,
    
    -- ─── Payment Amount Features ───
    ROUND(AVG(amount_paid)::NUMERIC, 2) AS avg_payment_amount,
    ROUND(SUM(amount_paid)::NUMERIC, 2) AS total_amount_paid,
    
    -- Payment-to-due ratio (are they paying in full?)
    -- WHY? Consistently paying 80% of due = chronic under-payer = risk
    ROUND(
        AVG(CASE WHEN amount_due > 0 THEN amount_paid / amount_due ELSE 1 END)::NUMERIC, 4
    ) AS avg_payment_to_due_ratio,
    
    -- ─── Recency Features ───
    -- WHY? Recent behavior is more predictive than historical.
    -- A late payment 3 months ago > a late payment 3 years ago.
    MAX(payment_date) AS last_payment_date,
    MIN(CASE WHEN is_late THEN due_date END) AS first_late_date,
    MAX(CASE WHEN is_late THEN due_date END) AS last_late_date,
    
    -- ─── Trend Features (last 3 and 6 installments) ───
    -- Computed via subquery below for recent windows
    
    -- ─── Payment Method Features ───
    -- WHY? Auto-debit customers default less because payment is automated.
    -- Payment method distribution reveals financial sophistication.
    COUNT(CASE WHEN payment_method = 'Auto-Debit' THEN 1 END) AS auto_debit_count,
    ROUND(
        COUNT(CASE WHEN payment_method = 'Auto-Debit' THEN 1 END)::NUMERIC / NULLIF(COUNT(*), 0), 4
    ) AS auto_debit_ratio
    
FROM repayment_history
GROUP BY loan_id, customer_id;


-- ──────────────────────────────────────────
-- D2. RECENT BEHAVIOR WINDOW FEATURES
-- ──────────────────────────────────────────
-- WHY WINDOW FEATURES? Credit risk is time-sensitive.
-- A borrower who was perfect for 2 years but started missing
-- payments recently is MORE risky than one who missed early
-- payments but has been clean for 2 years.

CREATE OR REPLACE VIEW v_recent_repayment_features AS
WITH ranked_payments AS (
    SELECT 
        loan_id,
        customer_id,
        installment_number,
        days_past_due,
        is_late,
        is_partial,
        amount_paid,
        amount_due,
        -- Rank from most recent
        ROW_NUMBER() OVER (PARTITION BY loan_id ORDER BY installment_number DESC) AS recency_rank
    FROM repayment_history
    WHERE payment_date IS NOT NULL  -- only count actual payments
)
SELECT 
    loan_id,
    customer_id,
    
    -- Last 3 months behavior
    ROUND(AVG(CASE WHEN recency_rank <= 3 THEN days_past_due END)::NUMERIC, 2) 
        AS avg_dpd_last_3,
    COUNT(CASE WHEN recency_rank <= 3 AND is_late THEN 1 END) 
        AS late_count_last_3,
    
    -- Last 6 months behavior
    ROUND(AVG(CASE WHEN recency_rank <= 6 THEN days_past_due END)::NUMERIC, 2) 
        AS avg_dpd_last_6,
    COUNT(CASE WHEN recency_rank <= 6 AND is_late THEN 1 END) 
        AS late_count_last_6,
    
    -- Last 12 months behavior
    ROUND(AVG(CASE WHEN recency_rank <= 12 THEN days_past_due END)::NUMERIC, 2) 
        AS avg_dpd_last_12,
    COUNT(CASE WHEN recency_rank <= 12 AND is_late THEN 1 END) 
        AS late_count_last_12,
    
    -- ─── DPD TREND (is behavior improving or worsening?) ───
    -- WHY? This is a KILLER feature in interviews.
    -- Positive trend = worsening, Negative trend = improving
    ROUND(
        AVG(CASE WHEN recency_rank <= 3 THEN days_past_due END) -
        AVG(CASE WHEN recency_rank BETWEEN 4 AND 6 THEN days_past_due END)
    , 2) AS dpd_trend_3v6,
    
    -- Consecutive late payments (streak detection)
    -- WHY? 3 consecutive lates is MUCH worse than 3 scattered lates.
    -- Consecutive = systematic failure, scattered = occasional issue.
    MAX(consecutive_late) AS max_consecutive_late_payments
    
FROM ranked_payments rp
LEFT JOIN LATERAL (
    SELECT COUNT(*) AS consecutive_late
    FROM (
        SELECT is_late,
               installment_number - ROW_NUMBER() OVER (ORDER BY installment_number) AS grp
        FROM repayment_history rh2
        WHERE rh2.loan_id = rp.loan_id AND rh2.is_late = TRUE
    ) streaks
    GROUP BY grp
    ORDER BY COUNT(*) DESC
    LIMIT 1
) cs ON TRUE
GROUP BY loan_id, customer_id;


-- ──────────────────────────────────────────
-- E. CROSS-TABLE / INTERACTION FEATURES
-- ──────────────────────────────────────────
-- WHY? The magic happens when you COMBINE features across tables.
-- Individual features tell a partial story. Interactions tell the full one.

CREATE OR REPLACE VIEW v_customer_aggregate_features AS
SELECT 
    c.customer_id,
    
    -- ─── Multi-loan features ───
    -- WHY? Repeat borrowers behave differently than first-timers.
    COUNT(la.loan_id) AS total_loans,
    COUNT(CASE WHEN la.is_default THEN 1 END) AS past_defaults,
    
    -- Historical default rate for this customer
    -- WHY? Past behavior is the best predictor of future behavior.
    ROUND(
        COUNT(CASE WHEN la.is_default THEN 1 END)::NUMERIC / NULLIF(COUNT(la.loan_id), 0), 4
    ) AS historical_default_rate,
    
    -- Total outstanding across all loans
    SUM(la.loan_amount) AS total_loan_exposure,
    
    -- Total exposure to income ratio
    -- WHY? Multiple loans compound risk. 3 loans at 30% DTI each = 90% total
    CASE 
        WHEN c.annual_income > 0 
        THEN SUM(la.loan_amount) / c.annual_income
        ELSE NULL 
    END AS total_exposure_to_income,
    
    -- Average loan amount (spending pattern)
    ROUND(AVG(la.loan_amount)::NUMERIC, 2) AS avg_loan_amount,
    
    -- Loan diversity (number of different loan types)
    -- WHY? Someone with personal + credit card + auto loan = diversified borrower
    COUNT(DISTINCT la.loan_type) AS loan_type_diversity,
    
    -- Time since first loan (months)
    -- WHY? Longer relationship with credit system = more data = more reliable
    EXTRACT(MONTH FROM AGE(CURRENT_DATE, MIN(la.application_date)))::INTEGER 
        AS months_since_first_loan,
    
    -- Recent loan velocity (loans in last 12 months)
    -- WHY? Sudden spike in loan applications = financial distress signal
    COUNT(CASE WHEN la.application_date >= CURRENT_DATE - INTERVAL '12 months' THEN 1 END) 
        AS loans_last_12_months

FROM customers c
LEFT JOIN loan_applications la ON c.customer_id = la.customer_id
GROUP BY c.customer_id, c.annual_income;


-- ============================================================
-- F. FINAL FEATURE TABLE (ML-READY)
-- ============================================================
-- This is the query that produces the final dataset for modeling.
-- Each row = one loan application with all engineered features.
--
-- Interview Insight: In production, this query would be scheduled
-- as a daily/weekly job, materializing into a feature store
-- (like Feast, Tecton, or a simple materialized view).

CREATE OR REPLACE VIEW v_ml_feature_table AS
SELECT 
    -- ★ Target Variable
    lf.is_default,
    lf.loan_id,
    
    -- Demographic Features
    df.age,
    df.age_group,
    df.gender,
    df.marital_status,
    df.dependents,
    df.education,
    df.employment_type,
    df.annual_income,
    df.log_annual_income,
    df.employment_stability_ratio,
    df.income_per_dependent,
    df.city_tier,
    df.verification_score,
    
    -- Bureau Features
    bf.credit_score,
    bf.credit_score_tier,
    bf.total_accounts,
    bf.active_accounts,
    bf.active_account_ratio,
    bf.overdue_30_count,
    bf.overdue_60_count,
    bf.overdue_90_count,
    bf.delinquency_severity_score,
    bf.has_serious_delinquency,
    bf.credit_utilization,
    bf.utilization_bucket,
    bf.enquiries_last_6m,
    bf.enquiry_velocity,
    bf.credit_history_length_months,
    bf.is_thin_file,
    
    -- Loan Features
    lf.loan_amount,
    lf.log_loan_amount,
    lf.loan_term_months,
    lf.term_bucket,
    lf.interest_rate,
    lf.rate_tier,
    lf.emi_amount,
    lf.loan_type,
    lf.is_secured,
    lf.dti_ratio,
    lf.emi_to_income_ratio,
    lf.loan_to_income_ratio,
    lf.interest_to_principal_ratio,
    lf.effective_ltv,
    
    -- Repayment Behavior Features
    rf.total_installments_due,
    rf.late_payment_count,
    rf.late_payment_ratio,
    rf.partial_payment_count,
    rf.partial_payment_ratio,
    rf.missed_payment_count,
    rf.avg_dpd,
    rf.max_dpd,
    rf.std_dpd,
    rf.ever_90_plus_dpd,
    rf.avg_payment_to_due_ratio,
    rf.auto_debit_ratio,
    
    -- Recent Behavior
    rrf.avg_dpd_last_3,
    rrf.late_count_last_3,
    rrf.avg_dpd_last_6,
    rrf.late_count_last_6,
    rrf.dpd_trend_3v6,
    rrf.max_consecutive_late_payments,
    
    -- Customer Aggregate
    caf.total_loans,
    caf.past_defaults,
    caf.historical_default_rate,
    caf.total_exposure_to_income,
    caf.loan_type_diversity,
    caf.months_since_first_loan,
    caf.loans_last_12_months

FROM v_loan_features lf
LEFT JOIN v_demographic_features df ON lf.customer_id = df.customer_id
LEFT JOIN v_bureau_features bf ON lf.customer_id = bf.customer_id
LEFT JOIN v_repayment_features rf ON lf.loan_id = rf.loan_id
LEFT JOIN v_recent_repayment_features rrf ON lf.loan_id = rrf.loan_id
LEFT JOIN v_customer_aggregate_features caf ON lf.customer_id = caf.customer_id;


-- ============================================================
-- FEATURE COUNT SUMMARY
-- ============================================================
-- Total engineered features: ~65+
-- 
-- Category Breakdown:
-- ┌─────────────────────────┬───────┬──────────────────────────────┐
-- │ Category                │ Count │ Key Features                 │
-- ├─────────────────────────┼───────┼──────────────────────────────┤
-- │ Demographic             │  12   │ age_group, income ratios     │
-- │ Credit Bureau           │  16   │ delinquency score, velocity  │
-- │ Loan Characteristics    │  14   │ EMI ratios, rate tier        │
-- │ Repayment Behavioral    │  15   │ DPD stats, payment ratios    │
-- │ Recent Window           │   7   │ trend, consecutive streaks   │
-- │ Customer Aggregate      │   7   │ default rate, exposure       │
-- └─────────────────────────┴───────┴──────────────────────────────┘
--
-- Interview Tip: When asked "How many features did you engineer?",
-- don't just say "65". Say:
-- "I engineered ~65 features across 6 categories: demographic,
--  bureau, loan-level, behavioral, temporal windows, and customer
--  aggregates. The highest-signal groups were bureau delinquency
--  features and recent repayment behavior trends."
-- ============================================================
