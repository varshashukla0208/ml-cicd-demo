# CD Pipeline

The CD workflow publishes the FastAPI inference image to GitHub Container
Registry and can optionally deploy it to a Docker host over SSH.

## Required Repository Secrets For Remote Deploy

- `DEPLOY_HOST`: Remote server hostname or IP.
- `DEPLOY_USER`: SSH username.
- `DEPLOY_SSH_KEY`: Private SSH key with access to the server.

## Optional Repository Secrets

- `DEPLOY_PORT`: Host port for the API. Defaults to `8000`.
- `DEPLOY_CHECKPOINT_DIR`: Directory on the server containing `best_model.pth`.
  Defaults to `$HOME/image-classification-api/checkpoints`.
- `GHCR_TOKEN`: Token used by the remote server to pull private GHCR images.

## Runtime Model Path

The API loads the model from:

```text
/app/checkpoints/best_model.pth
```

On the remote server, place the trained checkpoint here by default:

```text
$HOME/image-classification-api/checkpoints/best_model.pth
```

or configure `DEPLOY_CHECKPOINT_DIR` to point to another directory.
