# Uplift Pricing Engine

**Who should get which discount, so a limited budget earns the most profit?**

A decision system for telecom retention. It goes beyond churn prediction: survival models estimate customer lifetime value, uplift models find who actually responds to an offer, and a budget-constrained optimizer picks one offer per customer to maximize expected profit. Competitor prices are scraped and fed in as market context.

Status: Phase 1
The base model using XGBoost and data load and 60/20/20 split; EDA; XGBoost churn model; Platt and isotonic calibration compared; MLflow logging; model saved to models/churn.joblib
