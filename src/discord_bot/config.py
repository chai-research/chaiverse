import os

from dotenv import load_dotenv


# Load environment variables from .env file for easy local development
load_dotenv()


APPLICATION_ID = int(os.getenv('APPLICATION_ID') or 0)
BOT_TOKEN = os.getenv('BOT_TOKEN')
DEVELOPER_KEY = os.getenv('DEVELOPER_KEY')
