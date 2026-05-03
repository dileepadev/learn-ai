---
title: AI for Financial Inclusion
description: Explore how artificial intelligence is expanding access to financial services for the unbanked and underbanked — through alternative credit scoring, mobile money platforms, fraud detection in emerging markets, microinsurance pricing, and responsible AI practices for fair lending in underserved communities.
---

Approximately **1.4 billion adults globally** lack access to a bank account, and billions more remain significantly underserved — unable to access credit, insurance, or savings products on fair terms. The primary barrier is not unwillingness to provide services but the **thin-file problem**: traditional financial institutions rely on credit bureaus and formal income documentation, data that simply does not exist for people who have spent their lives in the informal economy.

AI is disrupting this paradigm by identifying creditworthiness signals in unconventional data sources — mobile phone usage patterns, digital payment histories, social networks, and behavioral data — enabling accurate risk assessment for borrowers who have never had a formal financial relationship.

## The Thin-File Problem and Alternative Data

A traditional credit model uses variables like:

- Loan repayment history (FICO, credit bureau data)
- Length of credit history
- Current debt levels
- Types of credit accounts

For the 1.4 billion unbanked, none of this data exists. Alternative data sources that correlate with creditworthiness include:

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def build_alternative_credit_features(user_data: dict) -> pd.Series:
    """
    Engineer creditworthiness features from non-traditional data sources.
    
    These features are used by fintech lenders like Tala, Branch, Jumo,
    and Grab Financial to serve thin-file borrowers in emerging markets.
    
    All features must be:
    - Predictive (empirically validated correlation with repayment)
    - Legally permissible (jurisdiction-specific: some EU/US regulations prohibit
      use of social data or location)
    - Fair (not proxy for protected characteristics)
    """
    features = {}
    
    # ── Mobile phone behavior ──────────────────────────────────────────────
    phone = user_data.get("phone_data", {})
    
    # Call/SMS patterns: regularity suggests stable social network
    features["call_regularity_score"] = phone.get("coefficient_of_variation_calls", 0)
    features["unique_contacts_90d"] = phone.get("unique_contacts_90d", 0)
    features["night_call_ratio"] = phone.get("night_call_ratio", 0)   # high = irregular hours
    
    # Top-up behavior: frequent small top-ups = limited cash reserve planning
    topup = phone.get("airtime_topup", {})
    features["avg_topup_amount"] = topup.get("mean_amount", 0)
    features["topup_regularity"] = topup.get("days_between_topups_cv", 0)
    features["topup_count_30d"] = topup.get("count_30d", 0)
    
    # Device signals: phone age, model tier, charging pattern
    features["device_age_days"] = phone.get("device_age_days", 0)
    features["device_price_tier"] = phone.get("device_price_tier", 1)  # 1-5
    
    # ── Mobile money transactions ─────────────────────────────────────────
    mm = user_data.get("mobile_money", {})
    features["mm_account_age_days"] = mm.get("account_age_days", 0)
    features["monthly_inflow_mean"] = mm.get("monthly_inflow_mean", 0)
    features["monthly_inflow_cv"] = mm.get("monthly_inflow_cv", 0)    # income volatility
    features["merchant_payment_count_90d"] = mm.get("merchant_payments_90d", 0)
    features["bill_payment_regularity"] = mm.get("bill_payment_score", 0)
    features["savings_balance_mean"] = mm.get("savings_balance_mean", 0)
    
    # ── App behavior ──────────────────────────────────────────────────────
    app = user_data.get("app_behavior", {})
    features["app_open_frequency_daily"] = app.get("daily_opens", 0)
    features["time_in_app_daily_minutes"] = app.get("daily_minutes", 0)
    features["profile_completeness_pct"] = app.get("profile_pct", 0)
    features["location_count_90d"] = app.get("unique_locations_90d", 0)
    
    # ── Social graph signals ──────────────────────────────────────────────
    social = user_data.get("social", {})
    # Referral network quality: were friends who referred you good borrowers?
    features["referrer_repayment_rate"] = social.get("referrer_repayment_rate", 0.5)
    features["network_default_rate"] = social.get("network_default_rate", 0.1)
    
    return pd.Series(features)


class AlternativeCreditScorer:
    """
    Gradient boosting credit scoring model using alternative data.
    
    Trained on historical loan performance linked to mobile/behavioral features.
    Outputs:
    - Credit score (300-850 analog)
    - Loan approval recommendation
    - Suggested credit limit
    - Key risk factors for explainability
    """
    
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            min_samples_leaf=50,  # prevent overfitting on small demographic segments
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AlternativeCreditScorer":
        """y: 1 = repaid on time, 0 = defaulted"""
        self.feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict_creditworthiness(self, X: pd.DataFrame) -> pd.DataFrame:
        X_scaled = self.scaler.transform(X[self.feature_names])
        repay_prob = self.model.predict_proba(X_scaled)[:, 1]
        
        # Map repayment probability to 300-850 credit score range
        credit_scores = 300 + (repay_prob * 550).round().astype(int)
        
        return pd.DataFrame({
            "repayment_probability": repay_prob,
            "credit_score": credit_scores,
            "approved": repay_prob > 0.65,
            "suggested_limit_usd": self._estimate_limit(repay_prob, X)
        })

    def _estimate_limit(self, repay_prob: np.ndarray, X: pd.DataFrame) -> np.ndarray:
        """Suggest credit limit proportional to income and risk."""
        base_limit = X.get("monthly_inflow_mean", pd.Series(100))
        risk_multiplier = repay_prob * 2.5   # max 2.5× monthly income
        return (base_limit * risk_multiplier).clip(5, 500).round()
```

## Fairness and Bias in Alternative Credit Scoring

Alternative credit models carry significant fairness risks. Many seemingly neutral signals can be proxies for protected characteristics:

```python
from sklearn.inspection import permutation_importance
import shap

def audit_credit_model_fairness(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    protected_attributes: dict[str, pd.Series]
) -> pd.DataFrame:
    """
    Fairness audit for credit scoring models.
    
    Key metrics:
    - Demographic parity: approval rate equal across groups
    - Equalized odds: TPR and FPR equal across groups
    - Calibration: predicted probabilities match observed rates
    
    Regulators in many markets (US Equal Credit Opportunity Act, EU GDPR,
    Kenya's Data Protection Act) require explainability and non-discrimination.
    """
    results = []
    
    predictions = model.predict_creditworthiness(X_test)
    approval = predictions["approved"]
    
    overall_approval_rate = approval.mean()
    
    for attr_name, attr_values in protected_attributes.items():
        for group in attr_values.unique():
            mask = attr_values == group
            group_approval = approval[mask].mean()
            group_default_rate = (1 - y_test[mask]).mean()
            
            # Disparate impact: ratio of minority to majority approval rate
            disparate_impact = group_approval / overall_approval_rate
            
            results.append({
                "attribute": attr_name,
                "group": group,
                "approval_rate": round(group_approval, 3),
                "actual_default_rate": round(group_default_rate, 3),
                "disparate_impact_ratio": round(disparate_impact, 3),
                "flag": "REVIEW" if disparate_impact < 0.8 else "OK"
                # 80% rule: disparate impact < 0.8 triggers regulatory scrutiny
            })
    
    return pd.DataFrame(results)
```

## Mobile Money and Digital Payment Infrastructure

Mobile money platforms — **M-Pesa** (Kenya/Tanzania), **bKash** (Bangladesh), **GCash** (Philippines), **Paytm** (India) — have created the data infrastructure that makes alternative credit scoring possible. AI powers several critical functions:

**Fraud detection in real-time**: Mobile money transactions are particularly vulnerable to SIM swap fraud, account takeover, and social engineering. ML models score every transaction in under 100 milliseconds:

```python
def build_realtime_fraud_features(transaction: dict, account_history: dict) -> dict:
    """
    Real-time feature engineering for mobile money fraud detection.
    Must complete in <20ms to avoid transaction delay.
    
    Key signals for mobile money fraud:
    - Transaction velocity: many small transfers in quick succession
    - New recipient: transferring to an account first seen today
    - Device change: new phone/SIM associated with account
    - Unusual amount: relative to account history
    - Geographic anomaly: transaction from unusual location
    - Time anomaly: 3am transaction for dormant account
    """
    import time
    
    now = time.time()
    
    return {
        # Velocity features
        "txn_count_last_1h": account_history.get("count_1h", 0),
        "txn_count_last_24h": account_history.get("count_24h", 0),
        "amount_last_1h": account_history.get("amount_1h", 0),
        
        # Recipient risk
        "recipient_account_age_days": transaction.get("recipient_age_days", 0),
        "ever_transacted_recipient": transaction.get("prior_txn_count", 0) > 0,
        "recipient_fraud_score": transaction.get("recipient_risk_score", 0),
        
        # Account state
        "device_changed_last_7d": account_history.get("device_changed_7d", False),
        "sim_changed_last_7d": account_history.get("sim_changed_7d", False),
        "password_changed_last_24h": account_history.get("pwd_changed_24h", False),
        
        # Amount anomaly
        "amount_zscore_30d": (
            (transaction["amount"] - account_history.get("mean_amount_30d", 0))
            / (account_history.get("std_amount_30d", 1) + 1e-6)
        ),
        
        # Time features
        "hour_of_day": (now % 86400) / 3600,
        "is_dormant_account": account_history.get("days_since_last_txn", 0) > 30,
    }
```

## Microinsurance Pricing with ML

Traditional actuarial tables require years of loss history — data that does not exist for new insurance markets in developing countries. ML enables pricing based on proxy data:

- **Agricultural microinsurance**: Satellite-based crop monitoring (NDVI time series) combined with weather data enables index-based insurance payouts triggered by measurable environmental events — eliminating expensive claims adjustment
- **Health microinsurance**: Mobile health app usage (step counts, sleep patterns, medication reminders) proxies for health behaviors
- **Livestock insurance**: Satellite pasture condition monitoring for automated drought-triggered payouts

## Key Players and Models

| Company | Market | Model type | Key innovation |
| --- | --- | --- | --- |
| Tala | Kenya, India, Philippines | Alternative credit scoring | Mobile behavior features |
| Branch | Sub-Saharan Africa | ML credit + mobile wallet | Smartphone data access |
| Jumo | Africa, Asia | Credit marketplace | M-Pesa transaction history |
| Cignifi | Global | Telecom-based credit | Airtime purchase patterns |
| Grab Financial | Southeast Asia | Super-app credit | Ride-hailing + food delivery data |
| M-Kopa | East Africa | Asset financing | IoT device performance as collateral |

Financial inclusion AI carries responsibilities that exceed typical ML applications: errors in credit denial can lock families out of economic mobility for years. Models must be explainable enough to satisfy regulators and borrowers, audited for discriminatory impact, and designed with appeal processes for adverse decisions. The most successful deployments treat responsible AI not as a compliance burden but as a commercial necessity — unfair models generate regulatory risk and reputational damage that outweighs any short-term risk reduction.
