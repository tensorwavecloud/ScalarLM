import uvicorn


async def create_api(port, server_status):
    server_config = uvicorn.Config(
        "cray_infra.api.fastapi.main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(server_config)
    server_status.servers.append(server)

    await server.serve()
