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
7. **Metric:** RMSLE (root mean squared log error), to measure errors on a log scale and reduce the influence of very large visitor counts.

## Results

### Single validation window (26 Mar – 22 Apr 2017)
| Model | RMSLE (lower is better) |
|---|---|
| Restaurant average (all history) | 0.642 |
| Restaurant × weekday average (all history) | 0.564 |
| Restaurant × weekday, log-space, all history | 0.541 |
| Restaurant × weekday, log-space, last 26 weeks | 0.531 |
| Gradient boosting | 0.534 |

A simple weekday baseline using the last 26 weeks performs about as well as the gradient boosting model. In the model, each restaurant's weekday average has by far the largest permutation importance; the other features are small or near zero.

### Rolling backtest: six consecutive 28-day windows (6 Nov 2016 – 22 Apr 2017)
| Method | Average RMSLE | Best window | Worst window |
|---|---|---|---|
| Weekday average, last 4 weeks | 0.5965 | 0.5691 | 0.6422 |
| Weekday average, last 26 weeks | 0.5509 | 0.5267 | 0.6071 |
| Weekday average, all history | 0.5564 | 0.5330 | 0.6085 |
| Gradient boosting | 0.5523 | 0.5314 | 0.6023 |

The 26-week baseline and gradient boosting performed similarly: each won 3 of 6 windows, by margins between 0.001 and 0.008. The 4-week window is clearly worse in every window.

### Reservation experiment: six-window backtest

A new gradient boosting experiment trains on historical 28-day forecast snapshots. Each snapshot uses only visitor history and reservations available before its forecast date. Training snapshots must finish before the validation window begins.

Both snapshot models use the same training examples and settings. Only the second receives reservation features: booked visitors, booking count, and whether the restaurant has prior reservation records.

| Method | Mean RMSLE across six windows |
|---|---|
| 26-week weekday baseline | 0.5509 |
| Snapshot gradient boosting without reservations | 0.5459 |
| Snapshot gradient boosting with reservations | 0.5384 |

Adding reservation features improved RMSLE in all six windows, reducing the mean score by 0.0075. The reservation model beat the 26-week baseline in five of six windows.

The largest improvement from reservations occurred in December: 0.5962 to 0.5789. In March, the largest improvement was in the first forecast week, consistent with greater availability of near-term bookings.

These results are exploratory: the validation windows have already been used for analysis. The snapshot models also differ from the original gradient boosting setup in their training examples, inputs, and early-stopping settings.

## Key findings
- Weekends are busier, with Saturday the busiest day; weekdays are similar.
- The jump in total visitors around July 2016 coincides with more restaurants appearing in the data, so totals aren't a reliable demand measure.
- Across six windows, the 26-week weekday average was best or tied in every window and beat all-history averaging in all six (by 0.001–0.010). The 16-week average was within about 0.005 across these windows, so I wouldn't treat 26 weeks as special. The 4-week average was worst in every window.
- The orignal gradient boosting didn't beat the 26-week baseline: it won 3 of 6 windows and lost 3, averaging 0.5523 vs 0.5509. This is consistent with the model leaning almost entirely on each restaurant's weekday average.
- The December window was the hardest for every method. The 26-week baseline and gradient boosting scored about 0.61, compared with roughly 0.53–0.56 in the other windows. Errors concentrate around year-end (see the New Year finding below). Since the data starts in Jan 2016, no model has seen a previous year-end.
- A handful of restaurants have very large errors (RMSLE 1.1–1.9). The worst has no records from August 2016 to February 2017, and its average daily visitors fell from about 26 (June 2016) to about 12 (July 2016) to about 6 (March 2017), so a model using its full history predicts far too high. This is also why shorter averaging windows help. The 10 restaurants contributing the most total squared log error cover about 1% of validation rows but about 6.5% of total squared error, so they matter but don't drive the overall score. Excluding them lowers the gradient boosting RMSLE from 0.534 to 0.520, a small improvement, so they don't explain most of the model's error.
- The holiday and month features showed no importance, but the validation window contains no public holidays, so this isn't evidence that holidays don't matter.
- Around New Year, the number of restaurants with records falls from about 700 per day (late December) to 312 on 31 Dec and 147 on 1 Jan, then recovers to about 600 by 5 Jan. The early-January dip in total visitors coincides with a temporary drop in reporting restaurants, consistent with holiday closures, not a permanent decline. Errors peak on these days: the model over-predicts on 31 Dec and under-predicts on 22, 28 and 29 Dec.

## Limitations
- The backtest covers about six months (Nov 2016 – Apr 2017) and includes only one year-end period, so I can't say how the methods behave over a full year or across multiple years.
- The data starts in Jan 2016, so no model has seen a previous year-end.
- The 26-week window was selected using the same windows I report on, so its scores are slightly optimistic.
- Differences of less than about 0.01 RMSLE between methods are small relative to window-to-window variation, so I don't claim any method is better than another at that level.
- Restaurant statistics include each training row's own value, which slightly flatters the training fit. Training-row restaurant statistics include the row’s own target and later observations within the training period. Validation targets are excluded, but training features do not fully reproduce information available at historical forecast dates.
- Restaurants missing from a day's records are assumed closed or not reporting; the data can't confirm which.

## Next steps
- Test features that capture recent trends (last 7/28 days) and year-end effects, to see if the model can beat the baseline.
- Collect a longer history so that several year-ends are available.
- Detect gaps and level shifts per restaurant and down-weight old data.
- Handle restaurants with gaps or very short histories.
- Build historical forecast snapshots and test reservation features using only information available at each forecast date.
- Deploy a small Streamlit app.

## How to run
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Download the data into `data/raw/`, then run `notebooks/01_explore.ipynb`.