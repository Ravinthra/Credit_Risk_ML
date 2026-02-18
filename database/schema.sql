-- ============================================================
-- CREDIT RISK / LOAN DEFAULT PREDICTION SYSTEM
-- PostgreSQL Database Schema
-- ============================================================
-- 
-- WHY THIS SCHEMA DESIGN?
-- ────────────────────────
-- In real fintech systems, credit risk data lives across multiple
-- normalized tables. This mirrors production systems at banks/NBFCs:
--
-- 1. customers        → Demographics + employment (KYC data)
-- 2. credit_history   → Bureau data (CIBIL/Experian equivalent)
-- 3. loan_applications → The actual loan request + decision
-- 4. repayment_history → Monthly EMI payment behavior
--
-- Interview Insight: Interviewers love when you show awareness
-- of how data is actually stored vs. how it's consumed by ML models.
-- The gap between these two is where SQL feature engineering shines.
-- ============================================================


-- ──────────────────────────────────────────
-- TABLE 1: customers
-- ──────────────────────────────────────────
-- WHY? Contains static/slow-changing customer attributes.
-- In production, this maps to your KYC (Know Your Customer) data.
-- These features capture the borrower's demographic risk profile.

CREATE TABLE IF NOT EXISTS customers (
    customer_id         SERIAL PRIMARY KEY,
    
    -- Demographics
    age                 INTEGER NOT NULL CHECK (age BETWEEN 18 AND 80),
    gender              VARCHAR(10) CHECK (gender IN ('Male', 'Female', 'Other')),
    marital_status      VARCHAR(15) CHECK (marital_status IN ('Single', 'Married', 'Divorced', 'Widowed')),
    dependents          INTEGER DEFAULT 0 CHECK (dependents BETWEEN 0 AND 10),
    education           VARCHAR(20) CHECK (education IN ('High School', 'Bachelor', 'Master', 'PhD', 'Other')),
    
    -- Employment & Income
    -- WHY separate fields? Income alone is misleading without employment context.
    -- A ₹20L salary from TCS vs. freelance income have very different risk profiles.
    employment_type     VARCHAR(20) CHECK (employment_type IN ('Salaried', 'Self-Employed', 'Business', 'Freelance', 'Unemployed')),
    years_employed      NUMERIC(4,1) DEFAULT 0,          -- tenure at current job
    annual_income       NUMERIC(12,2) NOT NULL,           -- in local currency
    monthly_income      NUMERIC(10,2) GENERATED ALWAYS AS (annual_income / 12) STORED,
    
    -- Location (risk varies by geography)
    city_tier           INTEGER CHECK (city_tier IN (1, 2, 3)),  -- Tier-1 = metro
    state               VARCHAR(50),
    
    -- Contact & Identity
    phone_verified      BOOLEAN DEFAULT FALSE,
    email_verified      BOOLEAN DEFAULT FALSE,
    
    -- Metadata
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- INDEX: We'll frequently join on customer_id and filter by income
CREATE INDEX idx_customers_income ON customers(annual_income);
CREATE INDEX idx_customers_employment ON customers(employment_type);


-- ──────────────────────────────────────────
-- TABLE 2: credit_history
-- ──────────────────────────────────────────
-- WHY? This simulates credit bureau data (CIBIL, Experian, TransUnion).
-- In reality, this comes from an API call to the bureau.
-- Credit history is THE most predictive feature set for default prediction.
--
-- Interview Insight: "What's the single most important feature for credit risk?"
-- Answer: Credit score / past repayment behavior. It encodes years of financial
-- behavior into a single signal. But raw score alone isn't enough — you need
-- granular bureau data to capture nuance.

CREATE TABLE IF NOT EXISTS credit_history (
    credit_id           SERIAL PRIMARY KEY,
    customer_id         INTEGER NOT NULL REFERENCES customers(customer_id),
    
    -- Bureau Score (300-900 scale, like CIBIL)
    credit_score        INTEGER CHECK (credit_score BETWEEN 300 AND 900),
    
    -- Account Summary
    total_accounts      INTEGER DEFAULT 0,       -- total credit lines ever
    active_accounts     INTEGER DEFAULT 0,        -- currently open
    closed_accounts     INTEGER DEFAULT 0,        -- settled/closed
    
    -- Delinquency History
    -- WHY granular buckets? DPD (Days Past Due) buckets are industry standard.
    -- A 30-DPD is a "soft" miss; 90+ DPD is a serious red flag.
    overdue_30_count    INTEGER DEFAULT 0,        -- 30-day past due instances
    overdue_60_count    INTEGER DEFAULT 0,        -- 60-day past due
    overdue_90_count    INTEGER DEFAULT 0,        -- 90+ day past due (NPA territory)
    
    -- Utilization
    -- WHY? High utilization = financial stress signal.
    -- Optimal is 30-40%. Above 80% is a strong default predictor.
    total_credit_limit  NUMERIC(12,2) DEFAULT 0,
    current_balance     NUMERIC(12,2) DEFAULT 0,
    credit_utilization  NUMERIC(5,4) GENERATED ALWAYS AS (
        CASE WHEN total_credit_limit > 0 
             THEN LEAST(current_balance / total_credit_limit, 1.0)
             ELSE 0 
        END
    ) STORED,
    
    -- Enquiry History
    -- WHY? Multiple hard enquiries in short time = desperation signal
    enquiries_last_6m   INTEGER DEFAULT 0,
    enquiries_last_12m  INTEGER DEFAULT 0,
    
    -- Oldest Account Age (months)
    credit_history_length_months INTEGER DEFAULT 0,
    
    -- Metadata
    bureau_pull_date    DATE DEFAULT CURRENT_DATE,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_credit_customer ON credit_history(customer_id);
CREATE INDEX idx_credit_score ON credit_history(credit_score);


-- ──────────────────────────────────────────
-- TABLE 3: loan_applications
-- ──────────────────────────────────────────
-- WHY? This is the core entity — one row per loan application.
-- The 'is_default' column is our TARGET VARIABLE for ML.
--
-- Interview Insight: Default definition matters! Industry standard:
-- - 90+ DPD (Days Past Due) = NPA = Default
-- - Some models use 60+ DPD for early warning systems
-- - The choice affects your positive class rate and model behavior

CREATE TABLE IF NOT EXISTS loan_applications (
    loan_id             SERIAL PRIMARY KEY,
    customer_id         INTEGER NOT NULL REFERENCES customers(customer_id),
    
    -- Loan Details
    loan_amount         NUMERIC(12,2) NOT NULL CHECK (loan_amount > 0),
    loan_term_months    INTEGER NOT NULL CHECK (loan_term_months BETWEEN 3 AND 360),
    interest_rate       NUMERIC(5,2) NOT NULL CHECK (interest_rate BETWEEN 1 AND 40),
    
    -- Calculated EMI (Equated Monthly Installment)
    -- WHY store it? Avoids recalculating in every query. In production,
    -- this is computed at application time and locked in.
    emi_amount          NUMERIC(10,2),
    
    -- Loan Type & Purpose
    -- WHY? Different loan types have wildly different default rates.
    -- Unsecured personal loans default at 5-8x the rate of home loans.
    loan_type           VARCHAR(20) CHECK (loan_type IN (
                            'Personal', 'Home', 'Auto', 'Education', 
                            'Business', 'Credit Card', 'Gold'
                        )),
    loan_purpose        VARCHAR(50),   -- free text: "home renovation", "medical emergency"
    
    -- Collateral (for secured loans)
    is_secured          BOOLEAN DEFAULT FALSE,
    collateral_value    NUMERIC(12,2) DEFAULT 0,
    
    -- Derived Ratios (critical for credit risk)
    -- WHY? These ratios are more predictive than raw amounts.
    -- A ₹50K EMI means nothing without knowing monthly income.
    dti_ratio           NUMERIC(5,4),  -- Debt-to-Income ratio
    ltv_ratio           NUMERIC(5,4),  -- Loan-to-Value ratio (secured loans)
    
    -- Application Metadata
    application_date    DATE NOT NULL DEFAULT CURRENT_DATE,
    approval_date       DATE,
    disbursement_date   DATE,
    
    -- Application Status
    status              VARCHAR(20) DEFAULT 'Pending' CHECK (status IN (
                            'Pending', 'Approved', 'Rejected', 'Disbursed',
                            'Active', 'Closed', 'Written-Off'
                        )),
    
    -- ★ TARGET VARIABLE ★
    -- This is what our ML model predicts!
    -- 1 = Defaulted (90+ DPD), 0 = Non-default
    is_default          BOOLEAN DEFAULT FALSE,
    
    -- Risk Grade (assigned by existing scoring model, if any)
    risk_grade          VARCHAR(5) CHECK (risk_grade IN ('A', 'B', 'C', 'D', 'E', 'F')),
    
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_loan_customer ON loan_applications(customer_id);
CREATE INDEX idx_loan_type ON loan_applications(loan_type);
CREATE INDEX idx_loan_status ON loan_applications(status);
CREATE INDEX idx_loan_default ON loan_applications(is_default);


-- ──────────────────────────────────────────
-- TABLE 4: repayment_history
-- ──────────────────────────────────────────
-- WHY? Granular monthly payment records. This is the behavioral data
-- that tells you HOW a borrower repays, not just IF they default.
--
-- Interview Insight: Behavioral features (payment patterns, partial
-- payments, early payments) are often MORE predictive than static
-- demographics. This is where you differentiate from junior candidates.

CREATE TABLE IF NOT EXISTS repayment_history (
    repayment_id        SERIAL PRIMARY KEY,
    loan_id             INTEGER NOT NULL REFERENCES loan_applications(loan_id),
    customer_id         INTEGER NOT NULL REFERENCES customers(customer_id),
    
    -- Payment Details
    installment_number  INTEGER NOT NULL,        -- which EMI (1, 2, 3...)
    due_date            DATE NOT NULL,
    payment_date        DATE,                    -- NULL = not yet paid
    
    -- Amounts
    amount_due          NUMERIC(10,2) NOT NULL,
    amount_paid         NUMERIC(10,2) DEFAULT 0,
    
    -- Payment Behavior
    -- WHY track these separately? They reveal behavioral patterns:
    -- - Consistent partial payments = cash flow issues
    -- - Occasional late but full payment = forgetfulness vs. financial stress
    -- - Early payments = financially healthy
    days_past_due       INTEGER DEFAULT 0,
    is_late             BOOLEAN DEFAULT FALSE,
    is_partial          BOOLEAN DEFAULT FALSE,
    
    -- Running Balance
    outstanding_balance NUMERIC(12,2) DEFAULT 0,
    principal_component NUMERIC(10,2) DEFAULT 0,
    interest_component  NUMERIC(10,2) DEFAULT 0,
    
    -- Payment Method (can be a feature — auto-debit customers default less)
    payment_method      VARCHAR(20) CHECK (payment_method IN (
                            'Auto-Debit', 'UPI', 'NEFT', 'Cash', 'Cheque', 'Other'
                        )),
    
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_repayment_loan ON repayment_history(loan_id);
CREATE INDEX idx_repayment_customer ON repayment_history(customer_id);
CREATE INDEX idx_repayment_dpd ON repayment_history(days_past_due);
CREATE INDEX idx_repayment_due_date ON repayment_history(due_date);


-- ──────────────────────────────────────────
-- SUMMARY VIEWS (convenient for analysis)
-- ──────────────────────────────────────────

-- View: Customer with latest credit info
CREATE OR REPLACE VIEW v_customer_credit AS
SELECT 
    c.*,
    ch.credit_score,
    ch.total_accounts,
    ch.active_accounts,
    ch.overdue_30_count,
    ch.overdue_60_count,
    ch.overdue_90_count,
    ch.credit_utilization,
    ch.enquiries_last_6m,
    ch.credit_history_length_months
FROM customers c
LEFT JOIN credit_history ch ON c.customer_id = ch.customer_id;

-- View: Loan application with customer context
CREATE OR REPLACE VIEW v_loan_full AS
SELECT 
    la.*,
    c.age,
    c.gender,
    c.education,
    c.employment_type,
    c.annual_income,
    c.monthly_income,
    c.city_tier,
    ch.credit_score,
    ch.credit_utilization,
    ch.overdue_90_count
FROM loan_applications la
JOIN customers c ON la.customer_id = c.customer_id
LEFT JOIN credit_history ch ON c.customer_id = ch.customer_id;
