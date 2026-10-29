import pandas as pd
from dataclasses import dataclass
from .params import CorpusSpec

""" Logic for computing date ranges from config values
Used within Loaders
"""

@dataclass(frozen = True)
class DateWindow:
    display_years: pd.Timestamp
    end: pd.Timestamp
    data_start: pd.Timestamp

def datewindow(
    display_years: int = CorpusSpec().years,
    baseline_quarters: int = CorpusSpec().baseline_quarters,
    today: pd.Timestamp = pd.Timestamp.today(),
):

    end           = today.to_period("Q").start_time
    display_start = end - pd.DateOffset(years = display_years)
    data_start = display_start - pd.DateOffset(months = baseline_quarters * 3)

    return DateWindow(
        display_start,
        end,
        data_start
    )

