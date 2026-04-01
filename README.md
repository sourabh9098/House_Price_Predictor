# House Price Prediction — Machine Learning Project

This project made me realize how much goes into building something that actually works. It is not just about writing a model. It is about understanding the data, making honest decisions, and building something you can stand behind.

---

## What This Project Does

Predicts the sale price of a house based on real property details — the size of the living area, construction quality, garage capacity, and neighborhood. You enter the details, the model gives you an estimated price.

The goal was simple: build something accurate, explainable, and usable by a real person — not just a notebook that scores well on a metric and never gets opened again.

---

## The Dataset

- **Source:** Ames Housing Dataset
- **Records:** 2,930 house sales
- **Features used:** 51, including numerical, categorical, and one-hot encoded columns

The dataset is messy in the way real data always is — missing values, skewed distributions, categorical variables with too many levels. That made it worth working on.

---

## What I Built

### Data Preprocessing
- Handled missing values thoughtfully, not by just dropping rows
- Applied one-hot encoding to Neighborhood, House Style, Roof Style, and Sale Condition
- Scaled all numerical features using StandardScaler

### Features That Matter
Overall Quality, Living Area, Garage Capacity, Basement Area, 1st Floor Area, Bathrooms, Bedrooms, Neighborhood (26 types), House Style, Sale Condition, and Roof Style.

### Model
**Ridge Regression** — regularized, interpretable, and right-sized for this problem. It does not pretend to be more complex than it needs to be.

THE ACCURECY I GOT . R² score: 0.84 on test data
The model is stable and generalizes properly.


### Deployment
A full Streamlit web app so anyone can use the model without touching code. Real-time predictions, segment classification (Starter, Mid-Range, Premium, Luxury), and insight metrics like price per square foot.

---

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, Streamlit, Joblib


## Results

Algorithm: Ridge Regression
Evaluation Metrics:
R² Score: 0.84
Adjusted R²: ~0.82
MAE: ~21,000

The model shows good generalization .

The model generalizes well across different neighborhoods and house types. It is not perfect — no house price model ever is — but it is honest about what it knows.


---

## What I Learned

I started thinking the modeling part would be the hardest. It was not. The hard part was deciding which features to keep, handling neighborhoods with very few data points, and building a UI that a non-technical person could use without getting confused.

Ridge Regression also deserves more respect than it gets. When the data is clean and features are well-selected, a regularized linear model can be surprisingly powerful and far easier to explain to someone who is not a data scientist.

---

## About Me

I am a machine learning enthusiast who believes a model that can be deployed and understood is worth more than one that just scores well in a notebook. I built every part of this myself — from data cleaning to the Streamlit interface. I care about the full picture, not just the accuracy number.


## Contact

- Live Link - https://housepricepredictor-bysourabh.streamlit.app/
- GitHub: [your-username](https://github.com/sourabh9098)
- LinkedIn: [your-linkedin](www.linkedin.com/in/sourabh9098)


*Built with care. Not just code.*
