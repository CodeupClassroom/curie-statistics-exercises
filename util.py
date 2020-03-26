import itertools as it
import matplotlib.pyplot as plt
import pandas as pd
import env


def get_db_url(db):
    return f"mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}"


def get_expected_props(x1: pd.Series, x2: pd.Series) -> pd.DataFrame:
    """
    Returns expected proportions for the combination of 2 categorical variables.
    """
    counts = it.product(
        x1.value_counts(normalize=True).sort_index().iteritems(),
        x2.value_counts(normalize=True).sort_index().iteritems(),
    )

    # # This way is a little slower than the explicit for loop below.
    # # (~ 10ms vs 6ms for # the mpg data set using drv and trans)
    # return pd.concat([
    #     pd.Series(p_x1 * p_x2, index=pd.Index([(x1, x2)]))
    #     for (x1, p_x1), (x2, p_x2) in counts
    # ]).unstack()

    expected = pd.DataFrame()
    for (x1, p_x1), (x2, p_x2) in counts:
        expected.loc[x1, x2] = p_x1 * p_x2
    return expected


def get_expected_values(x1: pd.Series, x2: pd.Series) -> pd.DataFrame:
    """
    Returns expected values for the combination of 2 categorical variables.
    """
    return get_expected_props(x1, x2) * x1.shape[0]


def plot_group_proportions(
    df: pd.DataFrame, x1: str, x2: str, proportions=False, ax=None
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(13, 8))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    (
        df.groupby(x1)[x2]
        .apply(lambda s: s.value_counts(normalize=proportions))
        .unstack()
        .plot.bar(stacked=True, width=0.9, ax=ax)
    )
    ax.set(ylabel="proportion" if proportions else "count")
    ax.legend(title=x2)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    return ax
