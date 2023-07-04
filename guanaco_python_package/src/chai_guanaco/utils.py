import os


def guanaco_data_dir():
    home_dir = os.path.expanduser("~")
    data_dir = os.environ.get('GUANACO_DATA_DIR', f'{home_dir}/.chai-guanaco')
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def print_color(text, color):
    colors = {'blue': '\033[94m',
              'cyan': '\033[96m',
              'green': '\033[92m',
              'yellow': '\033[93m',
              'red': '\033[91m'}
    assert color in colors.keys()
    print(f'{colors[color]}{text}\033[0m')
