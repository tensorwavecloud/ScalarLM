import os
import shutil

def rmdir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"{path} has been deleted.")
    else:
        print(f"{path} does not exist.")
