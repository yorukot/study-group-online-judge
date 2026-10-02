# Cerulean Labs leaderboard UI

Build from this directory; no parent files are required:

```sh
docker build -t cerulean-judge-ui .
```

Or from the repository root:

```sh
docker build -t cerulean-judge-ui ./ui
```

The image builds React with Vite and serves it with Nginx on port 80.
Requests to `/api/` are forwarded to `JUDGE_API_URL` (default `http://api:8000`).
Set this to the backend's reachable HTTP or HTTPS origin, without a trailing slash or
path. The browser uses the same origin; W&B and Slack credentials stay on the API.

For a backend running locally on Docker Desktop or OrbStack:

```sh
docker network create cerulean-ui
docker run --rm --network cerulean-ui -p 8080:80 \
  -e JUDGE_API_URL=http://host.docker.internal:8000 \
  cerulean-judge-ui
```

Open http://localhost:8080. The backend must listen on an interface reachable
from Docker. On Linux, also pass `--add-host=host.docker.internal:host-gateway`.

For Compose, add this service to the judge's existing Compose file:

```yaml
  ui:
    build:
      context: ./ui
    ports:
      - "8080:80"
    environment:
      JUDGE_API_URL: http://api:8000
    restart: unless-stopped
```

`/healthz` checks the UI server, independently of backend availability.
Edit `src/lab-names.json` to rename labs, then rebuild the image.

HTTPS backends such as `JUDGE_API_URL=https://oj.cerulean.works` are supported;
Nginx sends the upstream hostname using TLS SNI. Rebuild and redeploy the UI
image after changing `nginx.conf.template`.

Use a user-defined Docker network (Compose creates one automatically) so the
embedded DNS resolver at `127.0.0.11` is available to the proxy.
