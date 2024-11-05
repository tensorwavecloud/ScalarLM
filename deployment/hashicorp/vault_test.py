import os

if 'MODAL_OPENAI_API_KEY' in os.environ:
    print("vault configured successfully")
else:
    print("vault not configured correctly")
    exit(1)
