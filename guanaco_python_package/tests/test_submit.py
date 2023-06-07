from guanaco_python_package.submit import submit_model


@pytest.fixture()
def mock_submission():
    submission = {
        "model_repo": "ChaiML/test_model",
        "developer_uid": "name",
        "developer_key": "mock-key",
        "generation_params": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 40,
            "repetition_penalty": 1.0,
        },
        "formatter": "PygmalionFormatter",
    }
    return submission


def test_submit_end_to_end(mock_submission):
    response = submit_model(mock_submission)
    expected_timestamp = "123456"
    expected_submission_id = f"name_{expected_timestamp}"
    assert response == {"submission_id": expected_submission_id}
