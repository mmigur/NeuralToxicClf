import os
from dotenv import load_dotenv

load_dotenv()

APP_TITLE = os.environ.get("APP_TITLE")
APP_VERSION = os.environ.get("APP_VERSION")
APP_DESCRIPTION = os.environ.get("APP_DESCRIPTION")

MODEL_PATH = os.environ.get("MODEL_PATH")

OK_STATUS = os.environ.get("OK_STATUS")