from datetime import datetime, timezone
import pytz
from typing import Literal


US_PACIFIC = pytz.timezone('US/Pacific')
UTC = pytz.timezone('UTC')


def is_epoch_time_in_date_range(epoch_time, date_range):
    start_epoch_time = _get_date_range_field(date_range, 'start_date') or 0
    end_epoch_time = _get_date_range_field(date_range, 'end_date') or float('inf')
    is_in = start_epoch_time < epoch_time < end_epoch_time
    return is_in


def us_pacific_string(date_string):
    return _create_date_string_in_timezone(date_string, US_PACIFIC)


def utc_string(date_string):
    return _create_date_string_in_timezone(date_string, UTC)


def convert_to_utc_iso_format(date_string=None, default_timezone=US_PACIFIC):
    date_string_in_utc = None
    if date_string is not None:
        date = datetime.fromisoformat(date_string)
        date = default_timezone.localize(date) if not date.tzinfo else date
        date_in_utc = date.astimezone(timezone.utc)
        date_string_in_utc = date_in_utc.isoformat()
    return date_string_in_utc


def _create_date_string_in_timezone(date_string, pytz_timezone):
    date = datetime.fromisoformat(date_string)
    date = pytz_timezone.localize(date)
    return date.isoformat()


def _get_date_range_field(date_range=None, field=Literal['start_date', 'end_date']):
    epoch_time = None
    if date_range and date_range.get(field):
        isoformat_date = date_range.get(field)
        date = datetime.fromisoformat(isoformat_date) 
        assert date.tzinfo
        epoch_time = date.timestamp()
    return epoch_time
