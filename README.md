# Indian Flight Fare Analysis and Prediction

Exploratory data analysis (EDA) and baseline regression models on 300,153 Indian domestic flight tickets, to understand what drives ticket prices and to predict the fare.

**Author:** Parth Dixit (Roll No. 28, LY CSE-AIML, Batch B2), Bharati Vidyapeeth (Deemed to be University), Kharghar, Navi Mumbai
**Programme:** Training Central - Artificial Intelligence (October 2026 Batch)

## Overview

An online travel agency wants to know how flight prices change with the airline, route, number of stops, travel class, duration, departure time and how early the ticket is booked. This project answers 54 analytical questions in Python and then trains regression models to predict the price.

**Main findings**

| Question | Finding |
|---|---|
| Biggest price driver | Travel class: Business median fare (Rs 53,164) is about 9x Economy (Rs 5,772) |
| Booking window | Economy fares in the last 3 days are 193% higher than 46-60 days ahead; the biggest drop is around 15 days before departure |
| Stops | Non-stop is cheapest (median Rs 4,499) vs one-stop (Rs 7,959) and two or more stops (Rs 8,307) |
| Airlines | Vistara is the most expensive (median Rs 15,543), AirAsia the cheapest (Rs 3,276) |
| Time of day | Morning departures cost the most (Rs 8,112), Late Night the least (Rs 4,499) |
| Duration | Weak to moderate link with price (Pearson 0.20, Spearman 0.32) |

**Prediction results** (80/20 train-test split)

| Model | MAE (Rs) | RMSE (Rs) | R2 |
|---|---|---|---|
| Random Forest | 1,103.78 | 2,715.75 | 0.99 |
| Hist Gradient Boosting | 2,156.35 | 3,673.18 | 0.97 |
| Linear Regression | 4,500.71 | 6,814.94 | 0.91 |

## Dataset

Kaggle - [Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction) (`Clean_Dataset.csv`): 300,153 rows and 11 columns (`airline`, `flight`, `source_city`, `departure_time`, `stops`, `arrival_time`, `destination_city`, `class`, `duration`, `days_left`, `price`). The CSV is not included in this repository; download it from Kaggle.

## Approach

1. **Data inspection and cleaning:** shape, types, missing values, duplicates, category consistency, price validity, data dictionary.
2. **Univariate analysis:** price statistics, histogram with KDE, IQR outliers, percentiles, and distributions of airline, city, stops, duration and days left.
3. **Airline, route and booking-window analysis:** group statistics, boxplots, pivot tables, heatmaps, engineered `route` and `booking_window` features.
4. **Duration, stops and time-of-day analysis:** Pearson and Spearman correlations, comparison by stops and by departure and arrival period, Economy and Business analysed separately.
5. **Prediction:** one-hot encoding, ordinal stops, 80/20 split, Linear Regression vs Random Forest vs Hist Gradient Boosting, feature importance.

## Selected visuals

| | |
|---|---|
| ![Price distribution](images/price_distribution.png) | ![Airline boxplot](images/airline_boxplot.png) |
| ![Booking window by class](images/booking_window_by_class.png) | ![Feature importance](images/feature_importance.png) |

## Repository structure

```
flight-fare-analysis/
├── README.md
├── requirements.txt
├── Indian_Flight_Fare_Analysis_and_Prediction.ipynb
├── images/        key plots used in this README
└── report/        project report (PDF or Word)
```

## How to run

**Google Colab or Kaggle:** upload the notebook, add `Clean_Dataset.csv` (or attach the Kaggle dataset), and run all cells.

**Locally:**

```bash
pip install -r requirements.txt
# put Clean_Dataset.csv in this folder (or in data/)
jupyter notebook Indian_Flight_Fare_Analysis_and_Prediction.ipynb
```

The notebook looks for the CSV in the Kaggle input folder, `data/` and the current folder. Random Forest training can take a few minutes on the full data.

## Tech stack

Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn.

## Limitations

- The data looks like sampled booking options from one point in time, with six airlines and six cities.
- Business class is sold by only two airlines (Vistara and Air India), so airlines must be compared within the same class.
- Seasonality and festival effects are not captured.
- Models were tested on a random split, so near-identical tickets can appear in both train and test sets; accuracy on unseen flights may be lower.



