import uvicorn


async def create_api(port, running_status):
    server_config = uvicorn.Config(
        "cray_infra.api.fastapi.main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
    server = uvicorn.Server(server_config)
    running_status.servers.append(server)

    await server.serve()
