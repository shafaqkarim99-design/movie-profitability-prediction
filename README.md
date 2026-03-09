# 🎬 Popcorn & Predictions: What Makes a Movie a Hit?

![Python](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square&logo=python)
![ML](https://img.shields.io/badge/Models-Random%20Forest%20%7C%20K--Means-orange?style=flat-square)
![Data](https://img.shields.io/badge/Data-Kaggle%20TMDB%2010K%20Movies-20BEFF?style=flat-square&logo=kaggle)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

## Overview

This group project investigates the key factors that drive movie profitability using a dataset of 10,000 popular films from Kaggle. Combining **unsupervised clustering** and **supervised machine learning**, the analysis identifies patterns in successful films and builds predictive models to forecast profitability using only pre-release features — information a studio would know before a film hits theatres.

> **Best model:** Random Forest Binary Classification achieved an **AUC of 0.74**, offering meaningful business value for profitability prediction using pre-release data alone.

---

## Research Questions

1. Can we predict a film's **profitability category** using only pre-release features?
2. What production and release features most strongly influence the **magnitude of profitability**?
3. Are there distinct **clusters of movies** that share similar success profiles?
4. How do **genre combinations and release timing** affect audience popularity trends?
5. Can we identify **underperforming high-budget films** and the patterns they share?

---

## Dataset

| Attribute | Detail |
|-----------|--------|
| **Source** | Kaggle — TMDB Movies Dataset |
| **Size** | 10,000 popular movies |
| **Original Features** | 15 |
| **Target Variable** | Revenue / Profitability |
| **Key Variables** | Budget, Revenue, Popularity, Genres, Runtime, Release Date |

**Data challenges addressed:**
- Missing values and zero budgets — filtered and imputed
- Nested genre lists — converted to binary one-hot features
- Pre-release films excluded — only fully released movies included
- Profitability skew dominated by outliers — handled in modelling

---

## Methodology

### Data Preprocessing (`data_preprocessing_and_cleaning_team_ac.py`)
A reusable preprocessing library built for the project, including functions for:
- Feature assessment (numeric and categorical distributions, outlier detection)
- Missing value imputation (mean, median, mode, arbitrary)
- Outlier treatment via Winsorization (IQR and z-score methods)
- Type conversion, dummy encoding, binning, and normalization
- Datetime feature extraction (year, quarter, month, season, weekend flags)
- Multi-valued list column conversion to binary features (genres, production companies)

### Unsupervised Learning — K-Means Clustering
Two distinct clusters identified:

| Cluster | Profile |
|---------|---------|
| **Cluster 0** | High-budget, high-revenue films; predominantly summer releases; stronger audience engagement |
| **Cluster 1** | Lower-budget, modest-revenue films; smaller audiences; limited distribution reach |

### Supervised Learning — Random Forest Models

Three models trained using **pre-release features only:**

| Model | Task | Key Result |
|-------|------|------------|
| **Random Forest Regression** | Predict continuous profitability | MAE: 3.13 |
| **Random Forest Multi-Class Classification** | Predict profitability tier (Major Loss → Highly Profitable) | Weighted F1: 0.37 |
| **Random Forest Binary Classification** | Predict profitable vs. not profitable | **AUC: 0.74** ✓ Best model |

### Top Predictive Features (Classification)
Budget, runtime, release day of year, release day, release year, genre count, major studio involvement, and genre type (drama, thriller, action, comedy) were the strongest predictors of the profitability category.

---

## Key Findings & Recommendations

| Insight | Recommended Action |
|---------|--------------------|
| **Budget** is the strongest profitability predictor | Prioritize higher-budget productions for stronger commercial returns |
| **Release timing** (year, week, day) is highly influential | Schedule launches during peak periods (summer, holidays) |
| **Runtime** ranked second in feature importance | Maintain genre-appropriate runtimes; avoid extremes |
| **Genre and studio** signal commercial viability | Align genre with proven market demand patterns |

---

## Repository Structure

```
├── data_preprocessing_and_cleaning_team_ac.py  # Reusable preprocessing library
├── Supervised_Learning_Model.html              # Supervised learning notebook (rendered)
├── PowerPoint_for_Milestone_3.pptx             # Final presentation deck
└── README.md
```

> **Note:** The Kaggle dataset is not included due to file size. Download it directly from [Kaggle TMDB Movies Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) and place it in the root directory before running.

---

## How to Run

> This notebook is shared for code review and methodology purposes. To replicate the analysis:

1. Download the TMDB dataset from Kaggle (link above)
2. Open `Supervised_Learning_Model.html` on GitHub to explore the full notebook inline — all outputs and visualisations are visible without running anything
3. To run interactively, convert the HTML back to `.ipynb` or request the original notebook file

**Required packages:**
```
pandas, numpy, matplotlib, seaborn, scikit-learn, feature-engine
```

---

## Limitations

- Post-release factors (marketing spend, competition, audience reception) not modelled
- Multi-class profitability classification performed modestly (F1: 0.37) — profitability categories are difficult to distinguish with pre-release data alone
- K-means clustering is sensitive to feature scaling and manual feature selection
- Historical patterns may not hold for streaming-first releases or emerging genres

---

## Collaborators

Developed as a group project at Ivey Business School, MSc Business Analytics program.

---

## Tools & Assistance

Development of preprocessing functions and model scripts was assisted by Claude (Anthropic) as an AI coding tool.

---

## Author

**Shafaq Karim** — Graduate in Business Analytics, Ivey Business School
[LinkedIn](https://www.linkedin.com/in/shafaqkarim/) · [Portfolio](https://shafaq-karim-t07p6o9.gamma.site/)
