# Classification-Model

In the highly competitive furniture & décor segment of Brazil’s largest online marketplace, understanding and pre-empting customer dissatisfaction is critical to driving repeat business and positive word-of-mouth. Our business case centers on predicting whether a customer’s five-point review (1–5) will fall into the “top-two-box” (scores of 4 or 5) versus lower scores. By converting the original Likert scale into a binary target (sat_t2b), we simplify reporting and create actionable signals for customer support and product teams.

Model Selection Rationale
We began with logistic regression as a transparent, fast baseline. Its coefficients deliver straightforward odds-ratio interpretations—showing, for example, how price or freight cost shifts the likelihood of satisfaction—while its built-in class‐weighting addresses our roughly 30/70 imbalance. However, linear boundaries limit its ability to capture interactions and non-linear effects across dozens of product subcategories and geographic segments.
To overcome this, we deployed a Balanced Random Forest within an SMOTE pipeline. SMOTE synthesizes additional “dissatisfied” samples during training, and the Balanced RF down-samples the majority class per tree, yielding robust, non-linear decision rules. This ensemble also provides built-in feature‐importance rankings to guide operational interventions.

Model Results

Metric	Logistic Regression	Balanced RF + SMOTE
ROC AUC	0.60	0.79
Accuracy	0.65	0.78
Macro-averaged F₁	0.58	0.72

Logistic Regression struggled with the minority “dissatisfied” class: precision 0.40, recall 0.43, F₁ 0.42, catching fewer than half of unhappy buyers with many false alarms.

Balanced RF + SMOTE markedly improved both classes: for dissatisfied customers, precision rose to 0.67 and recall to 0.52 (F₁ = 0.58); for satisfied buyers, precision/recall climbed to 0.82/0.89 (F₁ = 0.86). Overall discrimination jumped from near-random (AUC 0.60) to strong (AUC 0.79).

Key Recommendations

Targeted Outreach: Leverage model scores to auto-flag high-risk orders and trigger personalized follow-up—discounts, expedited support, or satisfaction surveys—tuning probability thresholds to balance resource costs against outreach precision (~0.67).

Operational Improvements: Focus on subcategories with below-par T2B rates (bulky furniture, complex assemblies). Pilot enhanced packaging, clearer assembly guides, or white-glove delivery options. Reevaluate freight-fee strategies, offering free or tiered shipping on at-risk SKUs.

Continuous Governance: Embed the SMOTE-RF pipeline into the CRM for real-time scoring, and retrain quarterly on new reviews. Monitor ROC AUC and class-wise metrics for drift, with automated alerts if performance degrades beyond a set threshold (e.g., AUC drop > 0.05).

Future Enhancements: Enrich features with text-sentiment from open comments, customer-lifetime metrics (order frequency, tenure), and test alternate samplers or boosting algorithms (e.g., XGBoost with scale_pos_weight). Use SHAP analyses to highlight individual-order risk drivers and refine tactical responses.
