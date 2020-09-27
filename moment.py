"""Calculate the integral of f(t)dt, tf(t)dt, and t**2f(t)dt."""
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s: %(levelname)s] %(message)s'
)


def integrate(y, x):
    """Calculate integral ydx.

    Parameters
    ----------
    y : np.ndarray, shape (1, n)
        y values
    x : np.ndarray, shape (1, n)
        x values

    Returns
    -------
    integral : float32
        integral value of ydx
    """
    assert y.size == x.size, "y and x should have the same size!"
    # dx is the width of each trapezoid
    dx = x[1:] - x[:-1]
    # yy is the height of each trapezoid
    # yy is essentialy the average of each adjacent pair of y
    yy = (y[1:] + y[:-1]) / 2
    # integral is the sum of all trapezoid area
    integral = np.sum(yy * dx)
    return integral


def preprocess_data(bg_fn, exp_fn, background_choice='separate',
                    background_H2=None, background_Ar=None, start_time=0.1):
    """Preprocess excel data.

    Subtracte background from experimental data.

    Parameters
    ----------
    bg_fn : xlsx
        background excel
    exp_fn : xlsx
        experimental data excel
    background_choice : str
        choose background source to calculate baseline
        choices: (separate, last_second, user_input)
    start_time : float
        start time of injection

    Returns
    -------
    values : np.ndarray, shape (m, n)
        experimental data after bg subtraction
    time : np.ndarray, shape (1, n)
        series of time
    index : np.ndarray, shape (1, n)
        index of injection during experiment
    temp : np.ndarry, shape (1, n)
        temperature
    """
    # read background excel
    log.info("Read background data: %s", bg_fn)
    df_bg = pd.read_excel(bg_fn, sheet_name=None)
    # get sheets
    # first 2 columns are meta data, remove them
    # last row is just one extra time, remove it
    df_bg1 = df_bg['1'].iloc[:-1, 2:]
    df_bg2 = df_bg['2'].iloc[:-1, 2:]
    # read experimental data excel
    log.info("Read experimental data: %s", exp_fn)
    df_exp = pd.read_excel(exp_fn, sheet_name=None)
    df_exp1 = df_exp['1'].iloc[:-1, 2:]
    df_exp2 = df_exp['2'].iloc[:-1, 2:]

    log.info("Preprocess")
    # get time, index and temperature
    time = df_exp1.iloc[1:, 0].astype(float)
    index = df_exp1.columns[1:].astype(int)
    temp = df_exp1.iloc[0, 1:].astype(float)

    # cut tail values that are background noise
    cut_tail = False
    # get the mean of each backround
    log.info("Background choice: %s", background_choice)
    if background_choice == 'separate':
        bg_means1 = df_bg1.iloc[1:, 1:].values.mean()
        bg_means2 = df_bg2.iloc[1:, 1:].values.mean()
        bg_cutoff1 = df_bg1.iloc[1:, 1:].values.max()
        bg_cutoff2 = df_bg2.iloc[1:, 1:].values.max()
        cut_tail = True
    elif background_choice == 'last_second':
        # search the starting row of the last second
        start = time.values[-1] - 1
        row = np.where(time.values >= start)[0][0]
        bg_means1 = df_exp1.iloc[row+1:, 1:].values.mean(axis=0)
        bg_means2 = df_exp2.iloc[row+1:, 1:].values.mean(axis=0)
        bg_cutoff1 = df_exp1.iloc[row+1:, 1:].values.max(axis=0)
        bg_cutoff2 = df_exp2.iloc[row+1:, 1:].values.max(axis=0)
        cut_tail = True
        df_exp1 = df_exp1.iloc[:row+1, :]
        df_exp2 = df_exp2.iloc[:row+1, :]
        time = time[:row]
    elif background_choice == 'user_input':
        if (not background_H2) or (not background_Ar):
            raise ValueError("set background values when choosing user_input")
        bg_means1 = background_Ar
        bg_means2 = background_H2
    else:
        log.warning("background choice not understood, "
                    "choose from (separate, last_second, user_input)")

    # subtract the mean from experimental data for each column
    log.info("Subtract background from experimental data")
    start_row = np.where(time.values >= start_time)[0][0]
    true_exp1 = df_exp1.iloc[start_row+1:, 1:] - bg_means1
    true_exp1 = true_exp1.values
    true_exp2 = df_exp2.iloc[start_row+1:, 1:] - bg_means2
    true_exp2 = true_exp2.values
    # shift time to the start time of injection
    time = time.values[start_row:]-start_time

    # cut tail
    if cut_tail:
        log.info("Cut tail")
        cut1 = bg_cutoff1 - bg_means1
        cut2 = bg_cutoff2 - bg_means2
        true_exp1[true_exp1 <= cut1] = 0
        true_exp2[true_exp2 <= cut2] = 0

    return (true_exp1, true_exp2,
            time, index, temp.values)


def process(bg_fn, exp_fn, out_fn=None,
            background_choice='separate',
            background_H2=None,
            background_Ar=None,
            start_time=0.1):
    """Process experimental data and calculate three integrals.

    Two steps apply for all injections:
       1. subtract background
       2. calculate the integral

    Parameters
    ----------
    bg_fn : xlsx
        background excel
    exp_fn : xlsx
        experimental data excel
    out_fn : str, optional
        path to result file
    """
    log.info("Start processing")
    y_Ar, y_H2, time, index, temp = preprocess_data(
        bg_fn, exp_fn, background_choice,
        background_H2, background_Ar, start_time
    )
    log.info("Calculate integrals")
    # integral of f(t)dt
    M0_Ar = np.array([
        integrate(y_Ar[:, i], time) for i in range(y_Ar.shape[1])
    ])
    M0_H2 = np.array([
        integrate(y_H2[:, i], time) for i in range(y_H2.shape[1])
    ])
    M0H2_to_M0Ar = M0_H2 / (M0_Ar + np.finfo(float).eps)
    # integral of t*f(t)dt
    M1_Ar = np.array([
        integrate(time*y_Ar[:, i], time) for i in range(y_Ar.shape[1])
    ])
    M1_H2 = np.array([
        integrate(time*y_H2[:, i], time) for i in range(y_H2.shape[1])
    ])
    # integral of t^2*f(t)df
    M2_Ar = np.array([
        integrate(time**2 * y_Ar[:, i], time) for i in range(y_Ar.shape[1])
    ])
    M2_H2 = np.array([
        integrate(time**2 * y_H2[:, i], time) for i in range(y_H2.shape[1])
    ])

    # write results to a output file
    columns = ['Index', 'Temperature', 'M0H2', 'M0Ar',
               'M1H2', 'M1Ar', 'M2H2', 'M2Ar', 'M0H2/M0Ar']
    values = np.vstack([index, temp, M0_H2, M0_Ar, M1_H2,
                        M1_Ar, M2_H2, M2_Ar, M0H2_to_M0Ar]).T
    df_result = pd.DataFrame(values, columns=columns)
    if out_fn is None:
        out_fn = 'results.xlsx'
    log.info("Write result to %s", out_fn)
    df_result.to_excel(out_fn, index=False)
    log.info("Complete")
    return df_result


if __name__ == '__main__':
    import sys
    args = sys.argv[1:]
    process(*args)
