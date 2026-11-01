# Restaurant Demand Forecasting

Predicting daily visitor counts for Japanese restaurants, comparing simple baselines with a gradient boosting model. Built as a learning project in Python (pandas, scikit-learn).

## Problem
Given a restaurant and a date, predict how many visitors it will have. Restaurants use forecasts like this to plan staffing and ordering. I got interested in this after running a campus food-ordering platform (NLX Eats) and seeing how uneven demand was across the day and across vendors.

## Data
[Recruit Restaurant Visitor Forecasting](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting) dataset from Kaggle: about 252,000 restaurant-day records, plus restaurant details (genre, area) and a Japanese holiday calendar. The data isn't included in this repo. Download it from Kaggle into `data/raw/`.

## Approach
1. **Exploration:** looked at weekly patterns, trends and data quality.
2. **Validation:** trained on the past and tested on the most recent 28 days, because shuffling a time series would leak future information.
3. **Baselines:** the restaurant's overall average, and its average for that day of the week.
4. **Model:** gradient boosting trained on calendar features, holiday flags, restaurant type and area, and each restaurant's historical averages.
5. **Metric:** RMSLE (root mean squared log error), so errors count as percentages and a few very busy days don't dominate.

## Results

| Model | RMSLE (lower is better) |
|---|---|
| Restaurant average | 0.642 |
| Restaurant × weekday average | 0.564 |
| Gradient boosting | 0.534 |

The model beats the best baseline by about 5%. [Add one sentence on why the improvement is modest: for example, the model can't yet see each restaurant's recent trend.]

## Key findings
- Weekends are busier, with Saturday the busiest day. Weekdays are fairly similar to each other.
- The jump in total visitors around July came from [more restaurants appearing in the data, not more customers per restaurant. Confirm this with your chart].
- The dip in early January [your explanation: New Year closures, or permanent closures. Say what the chart showed].
- [Add one finding from feature importance or error analysis once you've run those cells.]

## Limitations
- Only the last 28 days are used for validation, so the score may not hold in other periods.
- Restaurant-level averages are computed on the training period and include each row's own value, which slightly flatters the training fit.
- The model doesn't use recent trends or weather.

## Next steps
- Add recency features (each restaurant's last 7 and 28 days) without leaking future data.
- Try other models and compare.
- Deploy a small Streamlit app.

## How to run
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Download the data into `data/raw/`, then run `notebooks/01_explore.ipynb`.