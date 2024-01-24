from datetime import datetime, timezone
import pytest

from chaiverse.metrics.feedback_metrics import FeedbackMetrics


TIMESTAMP_0101 = int(datetime(2024, 1, 1, 0, 0, 0, 0, timezone.utc).timestamp())
TIMESTAMP_0103 = int(datetime(2024, 1, 3, 0, 0, 0, 0, timezone.utc).timestamp())
TIMESTAMP_0105 = int(datetime(2024, 1, 5, 0, 0, 0, 0, timezone.utc).timestamp())

UTC_STRING_0102 = '2024-01-02T00:00:00+00:00'
UTC_STRING_0104 = '2024-01-04T00:00:00+00:00'

DATE_RANGE = dict(start_date=UTC_STRING_0102, end_date=UTC_STRING_0104)

def test_filter_for_date_range():
    feedbacks = {
        'feedback': {
            f'feedback_{TIMESTAMP_0101}': dict(id=1),
            f'feedback_{TIMESTAMP_0103}': dict(id=2),
            f'feedback_{TIMESTAMP_0105}': dict(id=3),
        }
    }
    feedback_metrics = FeedbackMetrics(feedbacks)
    feedback_metrics.filter_for_date_range(DATE_RANGE)

    assert len(feedback_metrics.feedbacks) == 1
    assert feedback_metrics.feedbacks[0]['id'] == 2
