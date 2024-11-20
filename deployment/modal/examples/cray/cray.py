import modal

cray_image = modal.Image.from_dockerfile(
    "Dockerfile", context_mount=modal.Mount.from_local_dir(".", remote_path=".")
)

app = modal.App()

@app.function(image=cray_image)
@modal.asgi_app()
def fastapi_app():
    from cray_infra.api.fastapi.main import app as web_app
    return web_app
