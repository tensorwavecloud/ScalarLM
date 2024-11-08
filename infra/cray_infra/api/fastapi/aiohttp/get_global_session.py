import aiohttp

session = None


def get_global_session():
    global session
    if session is None:
        session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=100))
    return session
