"""Date utilities for AirAgent."""

import datetime
import pytz


def get_current_date_pst() -> datetime.datetime:
    """
    Returns the current date and time in PST timezone.
    
    Returns:
        datetime.datetime: Current PST datetime
    """
    pst = pytz.timezone('America/Los_Angeles')
    return datetime.datetime.now(pst) 