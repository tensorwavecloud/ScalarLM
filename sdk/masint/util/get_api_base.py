
import masint

def get_api_base():
    if masint.api_base is None:
        return "http://localhost:8000"

    return masint.api_base

