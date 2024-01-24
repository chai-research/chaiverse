from datetime import datetime, timezone
import pytz

import pytest

from chaiverse.lib import date_tools


def _get_utc_date(date_string) -> datetime:
    date = datetime.fromisoformat(date_string)
    date = pytz.utc.localize(date)
    return date


def _get_utc_date_string(date_string) -> str:
    date = _get_utc_date(date_string)
    utc_date_string = date.isoformat()
    return utc_date_string


DATE_STRING_NO_TIME_ZONE = datetime.fromisoformat('2024-01-01').isoformat()
UTC_STRING_0102 = _get_utc_date_string('2024-01-02')
UTC_STRING_0104 = _get_utc_date_string('2024-01-04')

@pytest.mark.parametrize(    
    'epoch_time, date_range, expected_is_in', [
        (_get_utc_date('2024-01-01').timestamp(), dict(start_date=UTC_STRING_0102, end_date=UTC_STRING_0104), False),
        (_get_utc_date('2024-01-03').timestamp(), dict(start_date=UTC_STRING_0102, end_date=UTC_STRING_0104), True),
        (_get_utc_date('2024-01-05').timestamp(), dict(start_date=UTC_STRING_0102, end_date=UTC_STRING_0104), False),
        (_get_utc_date('2024-01-01').timestamp(), dict(start_date=None, end_date=UTC_STRING_0104), True),
        (_get_utc_date('2024-01-05').timestamp(), dict(start_date=UTC_STRING_0102, end_date=None), True),
        (_get_utc_date('2024-01-05').timestamp(), None, True),
    ]
)
def test_is_epoch_time_in_date_range(epoch_time, date_range, expected_is_in):
    is_in = date_tools.is_epoch_time_in_date_range(epoch_time, date_range)
    assert is_in == expected_is_in


def test_convert_to_utc_iso_format_can_handle_none():
    assert date_tools.convert_to_utc_iso_format(None) == None


def test_convert_to_utc_iso_format_can_convert_one_date_to_utc():
    assert date_tools.convert_to_utc_iso_format('2024-01-01T23:00:00-01:00') == '2024-01-02T00:00:00+00:00'


def test_us_pacific_string():
    assert date_tools.us_pacific_string('2024-01-01') == '2024-01-01T00:00:00-08:00'
    assert date_tools.us_pacific_string('2024-01-01 01:23') == '2024-01-01T01:23:00-08:00'


def test_convert_to_utc_iso_format_will_assume_input_is_us_pacific():
    assert date_tools.convert_to_utc_iso_format('2024-01-01T16:00:00-08:00') == '2024-01-02T00:00:00+00:00'
    assert date_tools.convert_to_utc_iso_format('2024-01-01T16:00:00') == '2024-01-02T00:00:00+00:00'


def test_utc_string():
    assert date_tools.utc_string('2024-01-01') == '2024-01-01T00:00:00+00:00'
    assert date_tools.utc_string('2024-01-01 01:23') == '2024-01-01T01:23:00+00:00'


def test_get_date_range_field_will_raise_if_missing_time_zone():
    date_range = dict(start_date=DATE_STRING_NO_TIME_ZONE)
    with pytest.raises(AssertionError):
        assert date_tools._get_date_range_field(date_range, 'start_date')
    

def test_get_date_range_field_will_return_start_date():
    date_range = dict(start_date=UTC_STRING_0102)
    assert date_tools._get_date_range_field(date_range, 'start_date') > 0


def test_get_date_range_field_will_return_end_date():
    date_range = dict(end_date=UTC_STRING_0102)
    assert date_tools._get_date_range_field(date_range, 'end_date') > 0


def test_get_date_range_field_will_return_none():
    date_range = dict()
    assert date_tools._get_date_range_field(date_range, 'start_date') is None
    assert date_tools._get_date_range_field(date_range, 'end_date') is None
