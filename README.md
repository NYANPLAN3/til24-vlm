# py-api-template

Template for FastAPI-based API server. Features:

- Supports both CPU/GPU-accelerated setups automatically.
- Poetry for package management.
- Ruff for formatting & linting.
- VSCode debugging tasks.
- Other QoL packages.

Oh yeah, this template should work with the fancy "Dev Containers: Clone Repository
in Container Volume..." feature.

## Usage Instructions

- Replace all instances of `py-api-template` & `py_api_template`.
  - Tip: Rename the `py_api_template` folder first for auto-refactoring.

## Useful Commands

```sh
# The venv auto-activates, so these work.
poe prod # Launch "production" server.
poe dev # Launch debugging server, use VSCode's debug task instead by pressing F5.

# Building docker image for deployment.
docker build -f Dockerfile . -t example

# Running FastAPI app (with GPU),
docker run --rm --gpus all -p 3000:3000 example
```
