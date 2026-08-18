# Predicting Airbnb Guest Satisfaction in London

A machine learning case study testing whether NLP-derived review signals (sentiment scores and topic-model outputs) improve prediction of guest satisfaction over listing attributes alone.

**Key result:** adding review-derived features raised the best model's ROC-AUC from 0.70 (listing attributes only) to 0.80 (Random Forest, full feature set).

## What's in this repo

- `airbnb_guest_satisfaction_showcase.ipynb` — full analysis: data cleaning, EDA, and four classification models (Logistic Regression, ANN, Decision Tree, Random Forest) compared with and without review-derived features
- `requirements.txt` — Python dependencies
- `data/` — not included in this repo (see below); place the two source files here to run the notebook

## How to run it

1. Clone or download this repo
2. `pip install -r requirements.txt`
3. Download `London_Airbnb_Listings.xlsx` and `London_Airbnb_Reviews.csv` and place them in a `data/` folder at the repo root
4. Open `airbnb_guest_satisfaction_showcase.ipynb` in Jupyter and run all cells

## Data & citation

The dataset is sampled from the intermediate research outputs of:

> Chen, B., Anker, T., & Liang, X. (2025). Business continuity management in the sharing economy: Insights from Airbnb online reviews. *Tourism Management*, 107, 105067. https://doi.org/10.1016/j.tourman.2024.105067

Used here with the original researchers' permission for portfolio/educational purposes. Review-topic variables were generated using the original study's fine-tuned Structural Topic Model (STM); see Table 2 of the source article for topic-label interpretations.

*Originally developed as a graded project for CGMAI3003U — Machine Learning for Predictive Analytics in Business (CBS International Summer University Programme); adapted here for portfolio presentation.*

## License

Code is released under the MIT License (see `LICENSE`). The dataset is not covered by this license — see the citation note above.
