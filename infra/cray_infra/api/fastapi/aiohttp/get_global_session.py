import aiohttp

session = None


def get_global_session():
    global session
    if session is None:
        session = aiohttp.ClientSession()
    return session
