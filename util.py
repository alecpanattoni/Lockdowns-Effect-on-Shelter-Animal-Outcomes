import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import chisquare


def tvd(data, col, group_on, chosen_axis=1) -> float:
    """
    Calculates total variation distance of the distribution of values
    in col between the two groups of group_on. Assumes that col is
    categorical.

    Parameters
    -----------
    data: pd.DataFrame
        DataFrame with labels for the two groups
    col: str
        String name for column containing the data
    group_on: str
        String name for column containing the two group labels
    chosen_axis: int
        Axis to calculate TVD for
    Returns
    -----------
    tvd: float
        Test statistic (total variation distance of the distribution of
        values in col between the two groups of group_on)
    """
    if chosen_axis == 0:
        tvd = (
            data.pivot_table(index=col, columns=group_on, aggfunc="size", fill_value=0)
            .apply(lambda x: x / x.sum())
            .diff(axis=1)
            .iloc[:, -1]
            .abs()
            .sum()
            / data[group_on].unique().size
        )
    else:
        tvd = data.pivot_table(
            index=col, columns=group_on, aggfunc="size", fill_value=0
        ).apply(lambda x: x / x.sum(), axis=chosen_axis).diff(axis=0).iloc[
            -1
        ].abs().sum() / (
            data[group_on].unique().size - 1
        )
    return tvd


def calc_pvalue(dist, obs, operator="greater_equal") -> float:
    """
    Calculates the p-value for total variation distance.

    Parameters
    ----------
    dist: np.array
        Array of permutated (simulated) test statistics
    obs: float
        Observed test statistic
    operator: string
        string representing the comparison to be made
        options are:
            1. "greater_equal" (default)
            2. "greater"
            3. "less_than_equal"
            4. "less"

     Returns
    -----------
    p_value: float
        the p-value
    """
    if operator == "less":
        return np.mean(dist < obs)
    elif operator == "greater":
        return np.mean(dist > obs)
    elif operator == "less_than_equal":
        return np.mean(dist < obs)
    else:
        return np.mean(dist >= obs)


def permutation_test(data, col, group_on, test_stat, chosen_axis=0, n=1000) -> tuple:
    """
    Calculates a distribution of permuted test statistics and the observed
    test statistic resulting from permutation tests.


    Parameters
    -----------
    data: pd.DataFrame
        DataFrame with labels for the two groups
    col: str
        String name for column containing the data
    group_on: str
        String name for column containing the two group labels
    test_stat: function
        Function to generate test statistic
    chosen_axis: int
        If test stat is TVD, takes in axis to calculate TVD for.
    n: int (default = 100)
        Number of permutation tests to be run.

    Returns
    -----------
    stats: np.array
        Array of permutated (simulated) test statistics
    obs: float
        Observed test statistic
    """
    # calculate observed test statistic
    obs = test_stat(data, col, group_on)

    # permutation test
    stats = np.zeros(n)

    # create a copy dataframe to avoid overwriting original
    shuffled_data = data.copy()
    shuffled_col = shuffled_data[group_on].values
    for i in range(n):
        shuffled_col = np.random.permutation(shuffled_col)
        shuffled_data["shuffled"] = shuffled_col
        created_stat = test_stat(shuffled_data, col, "shuffled", chosen_axis)
        stats[i] = created_stat

    return stats, obs


def diff_in_means(data, col, group_col) -> float:
    """difference in means"""
    return data.groupby(group_col)[col].mean().diff().abs().iloc[-1]


def chisq_bootstrap(data, animal_type, outcome_type, sample_size=250, n=1000):
    """
    Takes in a animal and outcome to subset the data by, computing multiple
    chi-squared tests for the frequencies of outcome before and after COVID.

    Parameters
    -----------
    data: pd.DataFrame
        Data to load.
    animal_type: str
        Animal to subset for ("CAT" or "DOG").
    outcome_type: str
        Outcome type to subset for.
    sample_size: int (default = 250)
        Size of each sample when resampling. Used to balance for chi-squared.
    n: int (default = 1000)
        Number of bootstrapped chi-squared tests to be run.

    Returns
    -----------
    test_stats: np.array
        Array of bootstrapped (simulated) test statistics (chi-squared p-vals)
    conf_interval: tuple
        95% interval for the p-value
    median: float
        Median p-value from the chi-squared tests.
    """
    before = data.loc[
        (data["Type"] == animal_type) & (data["Outcome After COVID"] == False)
    ]
    after = data.loc[
        (data["Type"] == animal_type) & (data["Outcome After COVID"] == True)
    ]
    before_prop = before["Outcome Type"].value_counts(normalize=True)
    after_prop = after["Outcome Type"].value_counts(normalize=True)
    test_stats = np.zeros(n)
    for i in range(n):
        beforesamp = pd.Series(
            np.random.choice(
                a=before_prop.index, size=sample_size, p=before_prop.to_numpy()
            )
        ).value_counts()
        aftersamp = pd.Series(
            np.random.choice(
                a=after_prop.index, size=sample_size, p=after_prop.to_numpy()
            )
        ).value_counts()
        before_outcome = beforesamp.loc[outcome_type]
        before_vc = np.array([before_outcome, sample_size - before_outcome])
        after_outcome = aftersamp.loc[outcome_type]
        after_vc = np.array([after_outcome, sample_size - after_outcome])
        test_stats[i] = chisquare(before_vc, after_vc)[1]

    return (
        test_stats,
        np.percentile(a=test_stats, q=[2.5, 97.5]),
        np.median(test_stats),
    )


def quant_dist(data, N):
    """
    Takes in a series, and a number N and uniform randomly
    selects N number of values from a histogram generated based
    on the series.

    Parameters
    -----------
    data: pd.Series
        Series to take quantitative numbers from
    N: int
        Number of values to generate

    Returns
    -----------
    out: np.array
        N uniform randomly generated values based on data
    """
    data = data.dropna()
    p, bins = np.histogram(data, bins=5)
    p = p / p.sum()
    bin_width = np.diff(bins)[0]
    endpoints = np.random.choice(bins[:-1], p=p, size=N)
    out = np.array([np.random.uniform(x, x + bin_width) for x in endpoints]).astype(int)
    return out


def impute_portion(data):
    """
    Takes in a series and imputes missing values for that series.

    Parameters
    -----------
    data: pd.Series
        Series to be imputed

    Returns
    -----------
    data: pd.Series
        Series with imputed values
    """
    data = data.copy()
    N = data.isnull().sum()
    imputed = quant_dist(data, N)
    data[data.isnull()] = imputed
    return data


def plot_by_month(data, column, xlabel, ylabel, axis):
    """Takes in a dataframe and plots over time.

    Paramters
    ---------
    data: pd.DataFrame
        Data. Full shelter data or a subset.
    column: str
        Column name. Should have date information.
        Examples include 'Outcome Date', 'Intake Date'
    xlabel: str
        Label for x-axis
    ylabel: str
        Label for y-axis
    axis: int
        Axis number for multi-plotting.

    Returns
    -------
    None.

    Side Effects
    ------------
    a line plot
    """
    to_plot = data.copy()[[column]]
    to_plot["year"] = to_plot[column].dt.year
    to_plot["month"] = to_plot[column].dt.month
    plot = to_plot.groupby(["year", "month"]).count()[column].plot(kind="line", ax=axis)
    x_ticks = [0] + (
        (to_plot.groupby("year").count()["month"] / to_plot.shape[0]) * 100
    ).cumsum().loc[2014:2019].to_list()
    x_ticklabels = [2014, 2015, 2016, 2017, 2018, 2019, 2020]
    plot.set(xlabel=xlabel, ylabel=ylabel)
    plot.set_xticks(x_ticks)
    plot.set_xticklabels(x_ticklabels)
    return plot
