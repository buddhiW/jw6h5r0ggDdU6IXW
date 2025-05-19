# Term Deposit Marketing: A Two-Stage Optimization Pipeline

This project implements a strategic two-stage machine learning pipeline to predict customer subscriptions to a term deposit for a European banking institution. The primary aim is to enhance marketing efficiency by first broadly identifying potential leads with high recall, and then precisely targeting the most promising candidates to maximize conversion rates.

## Background

A startup specializing in ML solutions for the European banking market seeks to improve call center success rates for product subscriptions. This project designs a system that optimizes outreach and provides interpretable insights for strategic decision-making, moving beyond simple prediction.

## Data Description

The dataset originates from direct marketing campaigns for a term deposit product. It includes customer attributes (demographics, financial) and call-related information.

**Key Attributes:**
* `age`, `job`, `marital`, `education`, `default` (credit status)
* `balance` (average yearly balance)
* `housing` (housing loan), `loan` (personal loan)
* `contact` (communication type), `day`, `month` (last contact details)
* `duration` (last contact duration), `campaign` (number of contacts)

**Target Variable:**
* `y`: Has the client subscribed to a term deposit? (binary: yes/no). The dataset is highly imbalanced, with significantly fewer 'yes' instances.

## Goal(s) & Pipeline Strategy

* **Primary Goal:** Implement a two-stage pipeline to optimize marketing calls:
    1.  **Stage 1 (Model 1):** Maximize **recall** using pre-call data to filter out unlikely subscribers and reduce wasted call volume.
    2.  **Stage 2 (Model 2):** Maximize **precision** using all available data (including call interaction patterns) on the filtered list from Stage 1 to identify the best candidates for conversion.
* **Success Metrics:** Performance is evaluated using metrics suitable for imbalanced datasets, focusing on:
    * Model 1: Recall, and its impact on resource saving.
    * Model 2: Precision, F1-score, Balanced Accuracy, AUC-PR (precision-recall) on the filtered set.
    * (Note: Overall "accuracy" is not a primary focus due to severe class imbalance.)

## Methodology: Two-Stage Pipeline

The data is split (80% train, 20% test) for developing and evaluating the pipeline.

**Stage 1: Broad Candidate Identification (Model 1)**
1.  **Features Used:** Customer personal and financial attributes available *before* extensive calling (e.g., `age`, `job`, `balance`, `housing`).
2.  **Objective:** Maximize **recall** to capture a wide net of potential subscribers, thereby minimizing missed opportunities, while significantly reducing calls to clearly uninterested parties.
3.  **Training & Application:** Model 1 is trained on the training set. It then classifies the test set. Customers predicted as "yes" (potential subscribers) proceed to Stage 2. The remaining are deprioritized for this specific campaign.

**Stage 2: Precise Targeting (Model 2)**
1.  **Features Used:** All available features, including call-specific details (`duration`, `month`, `contact`) learned from historical campaign data, applied to the candidates passed from Model 1.
2.  **Objective:** Maximize **precision**. This model refines the list from Stage 1, identifying those most likely to convert.
3.  **Training & Application:** Model 2 is trained on the training set (using all features). It then classifies *only* the subset of the test data predicted as "yes" by Model 1.

**Common Technical Steps (explored in `main.ipynb`):**
* **Data Preprocessing:** Label Encoding, One-Hot Encoding, StandardScaler.
* **Handling Class Imbalance:** Crucial for both models. Techniques explored include SMOTE, SMOTEENN, and model-level class weighting (e.g., `scale_pos_weight`).
* **Model Selection:** Tree-based ensembles (RandomForest, XGBoost, LightGBM) are strong candidates.
* **Hyperparameter Tuning:** Optuna can optimize models for their stage-specific metrics.

## Results & Key Findings (Interpreted for the Two-Stage Pipeline)

This pipeline focuses on actionable outcomes rather than single, potentially misleading metrics like accuracy.

* **Stage 1 (Model 1 - Initial screening, optimizing call efficiency):**
    * Successfully identified a broader pool of potential leads while significantly reducing unproductive effort.
    * **Outcome:** This stage was able to **save an estimated 162.02 hours of unnecessary call time**, demonstrating a major improvement in operational efficiency.
* **Stage 2 (Model 2 - Focused targeting, lead Quality & conversion focus):**
    * Dramatically improved the concentration of genuine prospects from the filtered list.
    * **Outcome:** The ratio of **true positives to false positives among Model 2's "yes" predictions was almost 50:50**. This is a substantial improvement from the original dataset's approximate 7% "yes" rate, ensuring marketing efforts are highly targeted. This implies a precision of around 0.5 for this stage on the pre-filtered candidates.
    * The `main.ipynb` shows that component models (like a tuned LightGBM for Model 2 logic) can achieve strong F1-scores (e.g., ~0.62) and Balanced Accuracy (e.g., ~0.80) on their respective test sets, indicating robust predictive power.
* **Key Drivers for Subscription (from exploratory data analysis (EDA)):**
    *  `duration`: Last contact duration (most critical).
    *  `month`: Contacts in March or October.
    *  `balance`: Average yearly balance.
    *  `age`: Customer's age.
    *  `housing_yes`: Having a housing loan (negatively correlated).
* **Prioritized Customer Segments for Stage 2 Engagement:**
    * Customers filtered by Model 1 who then show characteristics indicative of higher engagement (e.g., likely to have longer call durations if contacted).
    * Those who can be targeted in historically successful months (March, October).
    * Older customers with higher financial balances.
    * Individuals without existing housing loans.

## Challenges Addressed

* **Resource drain from unproductive calls:** Directly mitigated by Model 1.
* **Low conversion rate from mass marketing:** Addressed by Model 2's precision targeting.
* **Identifying actionable behavioral drivers:** Revealed through EDA.

## Repository Contents

* `data/`: Contains the dataset used in the project.
* `main.ipynb`: Jupyter notebook containing model development.
* `main_backup.ipynb`: Contains more model tuning experiments.
* `EDA.ipynb`: Exoloratory data analysis.
* `README.md`: This file.
* `requirements.txt`: List of Python dependencies.

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/buddhiW/WuzwTDARz30dcTL4.git
    ```

2.  Create a virtual environment (recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    venv\Scripts\activate  # On Windows
    ```

3.  Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Open and run `main.ipynb` to reproduce the analysis.

To implement the full pipeline as described:
- Perform an 80/20 train/test split.
- **Train Model 1** (demographic/financial features, optimize recall) on the training set.
- Apply Model 1 to the test set to get a list of potential "yes" candidates and calculate call hours saved.
- **Train Model 2** (all features, optimize precision) on the training set.
- Apply Model 2 to Model 1's "yes" candidates from the test set.
- Evaluate Model 2's precision and TP:FP ratio on this final set.

## Conclusion Summary

The implemented two-stage pipeline provides a powerful, data-driven strategy for the bank's term deposit campaigns. Model 1 significantly cuts down on inefficient calls (saving ~162 hours), while Model 2 ensures that subsequent, more intensive efforts are focused on leads with a nearly 50:50 chance of being actual subscribers—a massive improvement in lead quality. This approach, underpinned by careful handling of class imbalance and analysis of key predictive features like call duration and contact timing, allows for more strategic, cost-effective, and successful marketing outcomes.