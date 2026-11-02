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
5. **Recency baselines:** weekday averages computed over the last 4 to 52 weeks, to test how much history to use.
6. **Diagnostics:** permutation feature importance and error breakdowns by weekday, holiday and restaurant.
7. **Metric:** RMSLE (root mean squared log error), so errors count as percentages and a few very busy days don't dominate.

## Results

| Model | RMSLE (lower is better) |
|---|---|
| Restaurant average (all history) | 0.642 |
| Restaurant × weekday average (all history) | 0.564 |
| Restaurant × weekday, log-space, all history | 0.541 |
| Restaurant × weekday, log-space, last 26 weeks | 0.531 |
| Gradient boosting | 0.534 |

A simple weekday baseline using the last 26 weeks performs about as well as the gradient boosting model. In the model, each restaurant's weekday average has by far the largest permutation importance; the other features are small or near zero.

## Key findings
- Weekends are busier, with Saturday the busiest day; weekdays are similar.
- The jump in total visitors around July 2016 coincides with more restaurants appearing in the data, so totals aren't a reliable demand measure.
- Averaging over roughly 16–26 weeks scored best (RMSLE 0.531–0.535), versus 0.574 for 4 weeks and 0.541 for all history: short windows are noisy, long ones include outdated behaviour. The differences between the best windows are small.
- A handful of restaurants have very large errors (RMSLE 1.1–1.9). The worst has no records from August 2016 to February 2017, and its average daily visitors fell from about 26 (June 2016) to about 12 (July 2016) to about 6 (March 2017), so a model using its full history predicts far too high. This is also why shorter averaging windows help. The 10 worst restaurants cover about 1% of validation rows but about 6.5% of total squared error, so they matter but don't drive the overall score. Excluding them lowers the gradient boosting RMSLE from 0.534 to 0.520, a small improvement, so they don't explain most of the model's error.
- The holiday and month features showed no importance, but the validation window contains no public holidays, so this isn't evidence that holidays don't matter.
- Around New Year, the number of restaurants with records falls from about 700 per day (late December) to 312 on 31 Dec and 147 on 1 Jan, then recovers to about 600 by 5 Jan. The early-January dip in total visitors is therefore a temporary holiday closure, not a permanent decline. Errors peak on these days: the model over-predicts on 31 Dec and under-predicts on 22, 28 and 29 Dec.

## Limitations
- Evaluation uses a single 28-day window (2017-03-26 to 2017-04-22) containing no public holidays, so holiday effects weren't tested.
- The best averaging window (26 weeks) was chosen using the same validation data, so its score is slightly optimistic.
- Differences of about 0.003 RMSLE between models are too small to distinguish with one validation window.
- Restaurant statistics include each training row's own value, which slightly flatters the training fit.

## Next steps
- Evaluate on several validation windows (rolling backtests), including ones with holidays.
- Add recency features (last 7 and 28 days) to the model.
- Detect gaps and level shifts per restaurant and down-weight old data.
- Handle restaurants with gaps or very short histories.
- Deploy a small Streamlit app.

## How to run
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Download the data into `data/raw/`, then run `notebooks/01_explore.ipynb`.