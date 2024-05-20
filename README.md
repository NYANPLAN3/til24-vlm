# til24-vlm

Template for FastAPI-based API server. Features:

- Supports both CPU/GPU-accelerated setups automatically.
- Poetry for package management.
- Ruff for formatting & linting.
- VSCode debugging tasks.
- Other QoL packages.

Oh yeah, this template should work with the fancy "Dev Containers: Clone Repository
in Container Volume..." feature.

Note: Competition uses port 5004 for VLM server.
Note: Compute Capability of T4 is 7.5, RTX 4070 Ti is 8.9.

## Useful Commands

The venv auto-activates, so these work.

```sh
# Launch debugging server, use VSCode's debug task instead by pressing F5.
poe dev
# Run test stolen from the official competition template repo.
poe test
# Building docker image for deployment, will also be tagged as latest.
poe build {insert_version_like_0.1.0}
# Run the latest image locally.
poe prod
# Publish the latest image to GCP artifact registry.
poe publish
```

Finally, to submit the image (must be done on GCP unfortunately).

```sh
gcloud ai models upload --region asia-southeast1 --display-name 'nyanplan3-vlm' --container-image-uri asia-southeast1-docker.pkg.dev/dsta-angelhack/repository-nyanplan3/nyanplan3-vlm:latest --container-health-route /health --container-predict-route /extract --container-ports 5004 --version-aliases default
```

## PyTorch Wheels

Err, use the NLP repo to build it. But change the build script to enable cuDNN first.
