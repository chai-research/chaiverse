import json


def test_file_wiped(chai, temp_dir, monkeypatch):
    monkeypatch.setenv('GUANACO_DATA_DIR', str(temp_dir))

    credentials_file = temp_dir / 'credentials.json'
    credentials_file.write_text(json.dumps({'api_key': 'bar'}))

    result = chai('logout')

    assert result.exit_code == 0, result.output
    assert not result.output

    assert not credentials_file.is_file()
