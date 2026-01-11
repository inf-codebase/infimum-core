from enum import Enum


CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

# API constants
API_V1 = "v1"

class CrawlingType(Enum):
    PDF = 1
    WEB = 2
    TEXT = 3

class Prompt:
    NEXT_CRYPTO_EVENT = """
    use time tool to find out current date and finally the main purpose find out the most important event in cryptocurrency next ?
    """
