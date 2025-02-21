# Docker builds

Check out prebuilt docker containers for different targets:

| Target | Container                   | Latest Release v0.5      |
-------- | --------------------------- | ------------------------ |
| NVIDIA | gdiamos/cray-nvidia:latest  | gdiamos/cray-nvidia:v0.5 |
| ARM    | gdiamos/cray-arm:latest     | gdiamos/cray-arm:v0.5    |
| AMD    | gdiamos/cray-amd:latest     | gdiamos/cray-amd:v0.5    |
| x86    | gdiamos/cray-cpu:latest     | gdiamos/cray-cpu:v0.5    |

For example, to launch a development server on a modern macbook, e.g. m2

```bash
docker run -it -p 8000:8000 --entrypoint /app/cray/scripts/start_one_server.sh gdiamos/cray-arm:v0.5
```
