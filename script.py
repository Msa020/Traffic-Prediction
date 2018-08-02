import numpy as np
import pandas as pd
from plotly.offline import plot
from statsmodels.tsa.seasonal import seasonal_decompose

def fetch_times(file_path='test_BdBKkAj.csv'):
  df = pd.read_csv(file_path)
  df['DateTime'] = pd.to_datetime(df.DateTime)
  return df.set_index('ID')

def fetch_data(file_path='train_aWnotuB.csv'):
  '''
    Loads the csv; parses datetimes; casts the data
    for each junction as a separate column (J1, ..., J4 for
    junctions 1 through 4); adds columns for hour of day,
    day of week, day of month and month; orders the data by
    time.
  '''
  data = pd.read_csv(file_path)
  data.set_index(pd.DatetimeIndex(data.DateTime), inplace=True)
  data = data.pivot(columns='Junction', values='Vehicles')
  data['hour'] = data.index.hour
  data['day_of_week'] = data.index.dayofweek
  data['day_of_month'] = data.index.day
  data['month'] = data.index.month
  data['year'] = data.index.year
  rename_dict = {jid: 'J' + str(jid) for jid in range(1, 5)}
  return data.rename(columns=rename_dict).sort_index()

def fit(data, name, with_plots=False):
  # plug all data into tsa to get daily cycle
  data = data.dropna()
  result = seasonal_decompose(data, model='additive')

  # replug trend and residual into to get weekly cycle
  new_observed = result.trend + result.resid
  new_observed = new_observed.dropna()
  dhrs = 24
  dsow = 7
  new_result = seasonal_decompose(new_observed,
                                  model='additive',
                                  freq=dhrs * dsow)

  # decompositions
  daily_season = result.seasonal
  weekly_season = new_result.seasonal
  trend = new_result.trend
  resid = new_result.resid

  # fit trend linearly:
  z = np.polyfit(trend.dropna().index.astype(int), trend.dropna().values, 1)

  if with_plots:
    # plot prediction

    f = z[0] * trend.index.astype(int) + z[1]
    trend_fit = pd.Series(f, index=trend.index)

    prediction = daily_season + weekly_season + trend_fit
    plot([{'x': result.observed.index, 'y': result.observed},
          {'x': prediction.index, 'y': prediction}],
          filename='prediction_'+name+'.html')

    # plot decomposition
    plot([{'x': daily_season.index, 'y': daily_season},
          {'x': weekly_season.index, 'y': weekly_season},
          {'x': trend.index, 'y': trend},
          {'x': resid.index, 'y': resid}],
          filename='decomposition_'+name+'.html')

  return z, daily_season, weekly_season

def prepare_lookups(trend, daily_season, weekly_season):
  dsl = pd.DataFrame({'value': daily_season,
                      'hour': daily_season.index.hour}) \
                      .drop_duplicates().set_index('hour').value

  wsl = pd.DataFrame({'value': weekly_season,
                      'wd': weekly_season.index.dayofweek,
                      'hour': weekly_season.index.hour}) \
                      .drop_duplicates().set_index(['hour', 'wd']).value

  return trend, dsl, wsl

def predict(times, trend, daily_season, weekly_season):
  t = pd.DatetimeIndex(times)
  tr = trend[0] * t.astype(int) + trend[1]
  ds = daily_season.loc[t.hour]
  ws = weekly_season.loc[pd.MultiIndex.from_arrays([t.hour, t.dayofweek])]
  return pd.Series(tr.values + ds.values + ws.values,
                    index=times.index, name='Vehicles').round().astype(int)

if __name__ == '__main__':
  try:
    df = fetch_data()

    # only use last four months for Junction 1
    # only use last three months for Junction 2
    # only use last three months for Junction 3
    # use everything for Junction 4
    data = [df.J1[(df.year == 2017) & (df.month >= 2)],
            df.J2[(df.year == 2017) & (df.month >= 3)],
            df.J3[(df.year == 2017) & (df.month >= 3)],
            df.J4]

    labels = ['Junction_' + str(i) for i in range(1, 5)]

    fits = [prepare_lookups(*fit(d, label)) for d, label in zip(data, labels)]

    # Prediction
    skeleton = fetch_times()
    predictions = pd.concat([
      predict(skeleton.query('Junction == @i + 1').DateTime, *f).reset_index()
      for i, f in enumerate(fits)])

    predictions.to_csv('predictions.csv', index=False)

  except Exception as e:
    print(e)
