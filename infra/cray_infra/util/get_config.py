
from cray_infra.util.default_config import Config

def get_config():
    # Convert the pydantic model to a dictionary
    return Config().dict()





