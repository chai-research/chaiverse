__all__ = ["FeedbackMetrics"]


from collections import defaultdict

import numpy as np

from chaiverse.lib import date_tools
from chaiverse.metrics.conversation_metrics import ConversationMetrics


class FeedbackMetrics():
    def __init__(self, feedback_data):
        feedback_dict = feedback_data['feedback']
        feedback_dict = _insert_server_epoch_time(feedback_dict)
        self.feedbacks = list(feedback_dict.values())

    def filter_duplicated_uid(self):
        self.feedbacks = _filter_duplicated_uid_feedbacks(self.feedbacks)

    def filter_for_date_range(self, evaluation_date_range):
        self.feedbacks = [
            feedback for feedback in self.feedbacks
            if date_tools.is_epoch_time_in_date_range(feedback['server_epoch_time'], evaluation_date_range)
        ]

    def calc_metrics(self):
        metrics = {}
        if len(self.convo_metrics) > 0:
            metrics = {
                'mcl': self.mcl,
                'thumbs_up_ratio': self.thumbs_up_ratio,
                'thumbs_up_ratio_se': self.thumbs_up_ratio_se,
                'repetition': self.repetition_score,
                'total_feedback_count': self.total_feedback_count,
            }
        return metrics

    @property
    def convo_metrics(self):
        return [ConversationMetrics(feedback['messages']) for feedback in self.feedbacks]

    @property
    def thumbs_up_ratio(self):
        is_thumbs_up = [feedback['thumbs_up'] for feedback in self.feedbacks]
        thumbs_up = sum(is_thumbs_up)
        thumbs_up_ratio = np.nan if not thumbs_up else thumbs_up / len(is_thumbs_up)
        return thumbs_up_ratio

    @property
    def thumbs_up_ratio_se(self):
        num = self.thumbs_up_ratio * (1 - self.thumbs_up_ratio)
        denom = self.total_feedback_count**0.5
        se = np.nan if self.total_feedback_count < 2 else num / denom
        return se

    @property
    def total_feedback_count(self):
        return len(self.feedbacks)

    @property
    def mcl(self):
        return np.mean([m.mcl for m in self.convo_metrics])

    @property
    def repetition_score(self):
        scores = np.array([m.repetition_score for m in self.convo_metrics])
        is_public = np.array([feedback.get('public', True) for feedback in self.feedbacks])
        return np.nanmean(scores[is_public])


def _insert_server_epoch_time(feedback_dict):
    for feedback_id, feedback in feedback_dict.items():
        feedback['server_epoch_time'] = int(feedback_id.split('_')[-1])
    return feedback_dict


def _filter_duplicated_uid_feedbacks(feedbacks):
    user_feedbacks = defaultdict(list)
    for feedback in feedbacks:
        user_id = feedback["conversation_id"].split("_")[3]
        user_feedbacks[user_id].append(feedback)
    feedbacks = [metrics[0] for _, metrics in user_feedbacks.items()]
    return feedbacks


