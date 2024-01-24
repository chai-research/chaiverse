from chaiverse.metrics import conversation_metrics

def test_conversation_metrics():
    bot_sender_data = {'uid': '_bot_123'}
    user_sender_data = {'uid': 'XLQR6'}
    messages = [
        {'deleted': False, 'content': 'hi', 'sender': bot_sender_data},
        {'deleted': False, 'content': '123', 'sender': user_sender_data},
        {'deleted': True, 'content': 'bye', 'sender': bot_sender_data},
        {'deleted': True, 'content': 'bye~', 'sender': bot_sender_data},
        {'deleted': False, 'content': 'dont go!', 'sender': bot_sender_data},
        {'deleted': False, 'content': '123456', 'sender': user_sender_data},
        {'deleted': False, 'content': 'bye', 'sender': bot_sender_data},
    ]
    convo_metrics = conversation_metrics.ConversationMetrics(messages)
    assert convo_metrics.mcl == 5
    assert convo_metrics.repetition_score == 0.25


def test_get_repetition_score_is_one_if_all_responses_are_the_same():
    responses = ['Hi', 'Hi', 'hi']
    score = conversation_metrics.get_repetition_score(responses)
    assert score == 1.


def test_get_repetition_score_is_zero_if_all_responses_are_different():
    responses = ['Hi', 'Hey', 'How are you?']
    score = conversation_metrics.get_repetition_score(responses)
    assert score == 0.


def test_get_repetition_score_ignores_repetition():
    responses = ['Hi !', '...Hi', 'hi']
    score = conversation_metrics.get_repetition_score(responses)
    assert score == 1.


def test_get_repetition_score_handels_corrupt_responses():
    responses = ['! !', '...', '.']
    score = conversation_metrics.get_repetition_score(responses)
    assert score == 1.


def test_get_repetition_score_handels_semi_corrupt_responses():
    responses = ['Heya', '...', '...']
    score = conversation_metrics.get_repetition_score(responses)
    assert score == 0.5


def test_get_repetition_score():
    bad_responses = ['Hi! I am Tom', 'Hey! I am Val', 'Hi, im tOM']
    good_responses = ['Hey! I am Tom', 'Hello there, I am Val', 'Byee~~~']
    bad_score = conversation_metrics.get_repetition_score(bad_responses)
    good_score = conversation_metrics.get_repetition_score(good_responses)
    assert bad_score > good_score
