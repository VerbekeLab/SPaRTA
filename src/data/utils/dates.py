import numpy as np

def get_time_delta(step, time_type):
    if time_type == 'hours':
        delta_t = np.timedelta64(step, 'h')
    elif time_type == 'days':
        delta_t = np.timedelta64(step, 'D')
    elif time_type == 'weeks':
        delta_t = np.timedelta64(7 * step, 'D')
    elif time_type == 'months':
        delta_t = np.timedelta64(step, 'M')
    else:
        raise ValueError("Time type must be 'hours', 'days', 'weeks', or 'months'")
    return delta_t

def start_end_dates(start_date, end_date, delta_t, width_t):
    start_dates = []
    end_dates = []
    date_t = start_date + delta_t
    while date_t <= end_date:
        date_t_1 = date_t - width_t
        start_dates.append(date_t_1)
        end_dates.append(date_t)
        date_t += delta_t
    return start_dates, end_dates

def define_dates(transaction_dates, time_step=1, time_width=1, time_type='days'):
    start_date = transaction_dates.min()
    end_date = transaction_dates.max()
    delta_t = get_time_delta(time_step, time_type)
    width_t = get_time_delta(time_width, time_type)
    start_dates, end_dates = start_end_dates(start_date, end_date, delta_t, width_t)
    return start_dates, end_dates

def exponential_time_decay(time_stamp, end_date, days_echo=3):
    gamma = -np.log(0.01)/days_echo
    delta_t = np.datetime64(end_date) - np.datetime64(time_stamp)
    delta_t_days = delta_t.astype('timedelta64[s]').astype(int) / (3600*24)
    decay = np.exp(-gamma * delta_t_days)
    return decay