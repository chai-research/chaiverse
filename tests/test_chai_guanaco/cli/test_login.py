import json


def test_prompt_hidden(chai, helpers, temp_dir, monkeypatch):
    monkeypatch.setenv('GUANACO_DATA_DIR', str(temp_dir))

    credentials_file = temp_dir / 'credentials.json'
    credentials_file.write_text(json.dumps({'api_key': 'bar'}))

    result = chai('login', input='foo')

    assert result.exit_code == 0, result.output
    assert result.output == helpers.dedent(
        f"""
        Please enter your API key:{' '}
        """
    )

    assert json.loads(credentials_file.read_text(encoding='utf-8'))['api_key'] == 'foo'
