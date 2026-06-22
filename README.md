# Task 5: Personal Loan Acceptance Prediction

## 🎯 Task Objective
The primary objective of this project is to build an explanatory predictive modeling framework to determine which bank customers are highly likely to accept a personal loan or term deposit marketing offer. By utilizing historical demographic, financial, and campaign transaction footprints from the Bank Marketing Dataset (`bank-full.csv`), this project helps the bank move away from broad cold-calling strategies toward data-driven targeting. This maximizes consumer marketing conversion rates while dropping operational outbound calling costs.

---

## 🛠️ Your Approach & Methodology

The repository implements a structured, end-to-end data science and machine learning pipeline:

1. **Dataset Integrity Verification:**
   * Loaded the comprehensive `bank-full.csv` dataset consisting of 45,211 rows and 17 features using a semicolon (`;`) structural delimiter.
   * Conducted full schema audits via `df.info()` and verified target label properties, revealing a heavily imbalanced conversion baseline (~11.7% "yes" responses).

2. **Exploratory Data Analysis (EDA):**
   * Generated distribution and frequency profiles across target variables, continuous features (`age`), and categorical attributes (`job`, `marital`).
   * Evaluated direct interactions between customer job categories and success behaviors to isolate high-conversion corporate demographics.

3. **Data Preparation & Structural Preprocessing:**
   * Automated text-to-numeric transformations using `LabelEncoder` across all nominal characteristics while mapping the target variable `y` directly into explicit binary fields ($0$ for 'no', $1$ for 'yes').
   * Partitioned the feature space ($X$) from the target space ($y$) and executed a robust **80/20 train-test split** integrated with `stratify=y` to lock down equal target representation layers across datasets.
   * Standardized variables via `StandardScaler` to provide normal feature distributions required for gradient-descent stability in Logistic Regression.

4. **Classifier Optimization & Comparative Metrics:**
   * Trained and optimized two distinct frameworks: **Logistic Regression** (max interactions set to 1000) and a **Decision Tree Classifier** (depth limited to 5 to protect generalization capabilities and prevent overfitting).
   * Critically assessed performance metrics including precision, recall, and multi-dimensional confusion matrices to look past baseline accuracy biases.

5. **Business Value Feature Extraction:**
   * Extracted internal tree node splits (`model.feature_importances_`) to map and rank exactly which operational attributes carry the heaviest decision-making weight for loan conversions.

---

## 📊 Results and Insights

### 1. Model Diagnostic Performance Summary
While both models generated an identical baseline overall accuracy score of ~90%, evaluating localized class matrices reveals a clear winner:
* **The Accuracy Bias:** Due to the severe target class imbalance (~88% "no" values), general accuracy is highly misleading. A model guessing "no" across every single row would look highly accurate while generating zero business value.
* **Why Decision Tree Wins:** The Decision Tree Classifier drastically outperformed Logistic Regression by capturing a **Recall score of 40%** for true buyers compared to a low **23% Recall** from the regression model. Utilizing the Decision Tree enables the telemarketing unit to capture nearly double the actual loan conversions from the same customer list.

### 2. Strategic Feature Weights & Business Insights
Evaluating the feature coefficients sorted by predictive importance values yielded highly actionable bank operational strategies:

1. **The 'Golden Minute' (`duration`):** The duration of the phone call is overwhelmingly the strongest predictor of customer conversion. While this is monitored *during* active calls, sales agents must be trained with engaging call hooks in the first 30 seconds to prolong connectivity, since lengthier calls strongly correlate with customer acceptance.
2. **Leveraging Historical Loyalty (`poutcome`):** Customers who previously accepted a marketing product in past campaigns are highly receptive to new financing offers. Future campaign priorities should instantly place these historical leads at the top of the queue.
3. **Financial Capacity Liquidity (`housing`):** Individuals without active housing loans maintain a higher net disposable monthly income margin. These profiles have the financial flexibility to accept personal consumer credit lines compared to clients managing heavy real-estate debts.

---

## 📁 Repository File Structure
* `bank-full.csv` - Complete production source dataset containing 45,211 interactions.
* `task_5.ipynb` - Comment-supported Jupyter Notebook documenting all cleaning scripts, visual distributions, classification metrics, and importance plots.
* `README.md` - Executive summary documentation.
