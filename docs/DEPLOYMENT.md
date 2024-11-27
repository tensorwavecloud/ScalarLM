# SMI Platform Deployment Guide

## Prerequisites
- Docker
- Dockerhub account
- depot.dev account
- Modal account

## 1. Build Docker Image with depot.dev

### Create depot.dev Account
1. Visit https://depot.dev/ and create an account

### Build and Push Docker Image
```bash
depot build --platform linux/amd64 \
    --build-arg BASE_NAME=cpu \
    --build-arg VLLM_TARGET_DEVICE=cpu \
    -t your-dockerhub/smi-platform:latest \
    --shm-size=8g . --push
```

#### Authorization
- When prompted, click the authorization link (e.g., `https://depot.dev/orgs/qmkv-blah-blah`)

## 2. Deploy to Modal

### Set Up Modal
1. Create an account at https://modal.com
2. Join the smi-workspace
3. Install Modal Python package
   ```bash
   pip install modal
   ```
4. Authenticate Modal
   ```bash
   modal setup
   ```
   - This will open a web browser for authentication
   - If not, manually copy the provided URL

### Configure Deployment

#### Update Docker Image Credentials
In `deployment/modal/staging/cpu/cray.py`, update the image configuration:

```python
cray_image = (
    modal.Image.from_registry(
        "your-dockerhub/image-tag",
        secret=modal.Secret.from_dict(
            {
                "REGISTRY_USERNAME": "your-username",
                "REGISTRY_PASSWORD": "your-dockerhub-passwd",
            }
        ),
    )
    .pip_install("fastapi >= 0.107.0", "pydantic >= 2.9")
    .copy_local_file(
        local_path=local_config_path, 
        remote_path="/app/cray/cray-config.yaml"
    )
)
```

### Serve Deployment
```bash
modal serve deployment/modal/staging/cpu/cray.py
```

#### Expected Output
- Two web function URLs will be generated:
  1. FastAPI App URL
  2. VLLM App URL

### Update Deployment Configuration
1. Stop the previous `modal serve` command
2. Update `deployment/modal/staging/cpu/cpu-deployment.yaml`
   - Replace `api_url` with FastAPI App URL
   - Replace `vllm_api_url` with VLLM App URL

### Rerun Deployment
```bash
modal serve deployment/modal/staging/cpu/cray.py
```

## 3. Testing Deployment

### Health Check
```bash
PYTHONPATH=/path/to/smi-platform/sdk python test/deployment/health.py

{'api': 'up', 'vllm': 'up', 'all': 'up'}
```
- Expected output: `{'api': 'up', 'vllm': 'up', 'all': 'up'}`

### Generate Test
```bash
PYTHONPATH=/path/to/smi-platform/sdk python test/deployment/generate.py

{'responses': [' 0 + 0 = 0\nWhat is 0 + 0', ' 2. What is 1 + 1? 2. What is', ' What is 2 + 2? What is 2 + 2?', ' 6\nWhat is 3 + 3? 6\nWhat is']}
```
- Verifies text generation functionality

### Training Job Test
```bash
PYTHONPATH=/path/to/smi-platform/sdk python test/deployment/train.py

{'job_id': '1', 'status': 'QUEUED', 'message': 'Training job launched', 'dataset_id': '800fc373a8befb18cff123cca003c10b2744ca37b488541a751bbb026063072c', 'job_directory': '/app/cray/jobs/77607eb0e1c248bc36048e6c60023f46a50cbd895149adb17dc76edc33511d37', 'model_name': '77607eb0e1c248bc36048e6c60023f46a50cbd895149adb17dc76edc33511d37'}
```
- Launches and verifies a training job

## Troubleshooting
- Ensure all credentials and URLs are correctly configured
- Check network connectivity
- Verify Docker and Modal authentication

## Notes
- Replace placeholders like `your-dockerhub`, `your-username`, and paths with your actual values
- Keep credentials secure and do not commit them to version control