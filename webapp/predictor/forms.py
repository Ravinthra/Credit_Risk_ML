"""
Credit Risk ML -- Django Forms
===============================
Loan application form with validation.

WHY Django Forms (not raw HTML)?
- Built-in validation (server-side)
- CSRF protection
- Automatic error messages
- Clean data access via form.cleaned_data
- Reusable across views

Interview Insight:
"Why server-side validation when you have JS?"
-> "Client-side validation is UX -- it gives instant feedback.
   Server-side validation is SECURITY -- it can't be bypassed.
   Always validate on both sides. Never trust client input."
"""

from django import forms


class LoanApplicationForm(forms.Form):
    """
    Collects ~15 raw inputs from the user.
    Derived features (ratios, logs, buckets) are computed in the preprocessor.

    WHY these specific fields?
    - They map to the features our ML model was trained on
    - We only ask for fields a real loan applicant would provide
    - Behavioral features (DPD, late payments) have been removed from
      the model to prevent data leakage
    """

    # ── Section 1: Personal Information ──────────────────────

    age = forms.IntegerField(
        min_value=18, max_value=70,
        widget=forms.NumberInput(attrs={
            'class': 'form-input', 'placeholder': 'e.g. 28',
            'id': 'id_age',
        }),
        help_text='Applicant age (18-70)',
    )

    gender = forms.ChoiceField(
        choices=[
            ('Male', 'Male'), ('Female', 'Female'), ('Other', 'Other'),
        ],
        widget=forms.Select(attrs={'class': 'form-select', 'id': 'id_gender'}),
    )

    education = forms.ChoiceField(
        choices=[
            ('Below Secondary', 'Below Secondary'),
            ('Secondary', 'Secondary'),
            ('Higher Secondary', 'Higher Secondary'),
            ('Bachelor', 'Bachelor'),
            ('Master', 'Master'),
            ('PhD', 'PhD'),
        ],
        widget=forms.Select(attrs={'class': 'form-select', 'id': 'id_education'}),
    )

    marital_status = forms.ChoiceField(
        choices=[
            ('Single', 'Single'), ('Married', 'Married'),
            ('Divorced', 'Divorced'), ('Widowed', 'Widowed'),
        ],
        widget=forms.Select(attrs={'class': 'form-select', 'id': 'id_marital_status'}),
    )

    dependents = forms.IntegerField(
        min_value=0, max_value=10,
        initial=0,
        widget=forms.NumberInput(attrs={
            'class': 'form-input', 'placeholder': '0',
            'id': 'id_dependents',
        }),
    )

    # ── Section 2: Employment & Income ───────────────────────

    employment_type = forms.ChoiceField(
        choices=[
            ('Salaried', 'Salaried'),
            ('Self-Employed', 'Self-Employed'),
            ('Business', 'Business'),
            ('Freelance', 'Freelance'),
            ('Unemployed', 'Unemployed'),
        ],
        widget=forms.Select(attrs={'class': 'form-select', 'id': 'id_employment_type'}),
    )

    employment_years = forms.FloatField(
        min_value=0, max_value=40,
        widget=forms.NumberInput(attrs={
            'class': 'form-input', 'placeholder': 'e.g. 5',
            'step': '0.5',
            'id': 'id_employment_years',
        }),
        help_text='Years at current job',
    )

    annual_income = forms.FloatField(
        min_value=50000, max_value=50000000,
        widget=forms.NumberInput(attrs={
            'class': 'form-input', 'placeholder': 'e.g. 600000',
            'step': '10000',
            'id': 'id_annual_income',
        }),
        help_text='Annual income in INR',
    )

    # ── Section 3: Credit Profile ────────────────────────────

    credit_score = forms.IntegerField(
        min_value=300, max_value=900,
        widget=forms.NumberInput(attrs={
            'class': 'form-input', 'placeholder': 'e.g. 720',
            'id': 'id_credit_score',
        }),
        help_text='CIBIL score (300-900)',
    )

    credit_utilization = forms.FloatField(
        min_value=0, max_value=100,
        initial=40,
        widget=forms.NumberInput(attrs={
            'class': 'form-input', 'placeholder': 'e.g. 40',
            'step': '5',
            'id': 'id_credit_utilization',
        }),
        help_text='Credit utilization %',
    )

    existing_loans = forms.IntegerField(
        min_value=0, max_value=20,
        initial=1,
        widget=forms.NumberInput(attrs={
            'class': 'form-input', 'placeholder': 'e.g. 2',
            'id': 'id_existing_loans',
        }),
        help_text='Number of active loans',
    )

    # ── Section 4: Loan Details ──────────────────────────────

    loan_type = forms.ChoiceField(
        choices=[
            ('Personal', 'Personal Loan'),
            ('Home', 'Home Loan'),
            ('Auto', 'Auto Loan'),
            ('Education', 'Education Loan'),
            ('Business', 'Business Loan'),
            ('Gold', 'Gold Loan'),
            ('Credit Card', 'Credit Card'),
        ],
        widget=forms.Select(attrs={'class': 'form-select', 'id': 'id_loan_type'}),
    )

    loan_amount = forms.FloatField(
        min_value=10000, max_value=50000000,
        widget=forms.NumberInput(attrs={
            'class': 'form-input', 'placeholder': 'e.g. 500000',
            'step': '10000',
            'id': 'id_loan_amount',
        }),
        help_text='Loan amount in INR',
    )

    loan_term_months = forms.IntegerField(
        min_value=3, max_value=360,
        initial=36,
        widget=forms.NumberInput(attrs={
            'class': 'form-input', 'placeholder': 'e.g. 36',
            'id': 'id_loan_term_months',
        }),
        help_text='Loan tenure in months',
    )

    interest_rate = forms.FloatField(
        min_value=1, max_value=40,
        initial=12,
        widget=forms.NumberInput(attrs={
            'class': 'form-input', 'placeholder': 'e.g. 12.5',
            'step': '0.25',
            'id': 'id_interest_rate',
        }),
        help_text='Annual interest rate %',
    )
