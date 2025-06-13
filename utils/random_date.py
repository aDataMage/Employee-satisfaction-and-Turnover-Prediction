import pandas as pd
import numpy as np

def random_date(start, end):
    """Generate a random date between two datetime objects."""
    delta = end - start
    random_days = np.random.uniform(0, delta.days)
    return start + pd.to_timedelta(random_days, unit='D')