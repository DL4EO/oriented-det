import json
from pathlib import Path

from sanic import Sanic, Blueprint, response
from sanic_openapi import swagger_blueprint
from v1 import v1
from sanic.request import Request
from sanic.response import HTTPResponse

# Declare Sanic application
app = Sanic(__name__)

_APP_DIR = Path(__file__).resolve().parent
_DESC = _APP_DIR / "description.json"


def _load_openapi_metadata() -> None:
    """Fill Sanic OpenAPI config from description.json when present (publication-friendly)."""
    if not _DESC.is_file():
        return
    try:
        with open(_DESC, encoding="utf-8") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return
    title = meta.get("title")
    version = meta.get("version")
    desc = meta.get("description", "")
    org = meta.get("organization", "")
    email = meta.get("email", "")
    if title:
        app.config.API_TITLE = str(title)
    if version:
        app.config.API_VERSION = str(version)
    if desc:
        app.config.API_DESCRIPTION = str(desc)
    if org or email:
        app.config.API_TERMS_OF_SERVICE = str(org)
        app.config.API_CONTACT_EMAIL = str(email)


_load_openapi_metadata()

if getattr(app.config, "API_VERSION", None) in (None, "", "YOUR_API_VERSION"):
    app.config.API_VERSION = "0.1.0"
if getattr(app.config, "API_TITLE", None) in (None, "", "YOUR_API_TITLE"):
    app.config.API_TITLE = "oriented-det inference"
if getattr(app.config, "API_DESCRIPTION", None) in (None, "", "YOUR_API_DESCRIPTION"):
    app.config.API_DESCRIPTION = "Oriented object detection API (see /api/v1/describe)."
app.config.API_PRODUCES_CONTENT_TYPES = ["application/json", "application/geo+json", "image/jpeg", "image/png"]
if getattr(app.config, "API_TERMS_OF_SERVICE", None) in (None, "", "YOUR_API_LICENCE"):
    app.config.API_TERMS_OF_SERVICE = "Apache-2.0"
if getattr(app.config, "API_CONTACT_EMAIL", None) in (None, "", "YOUR_EMAIL"):
    app.config.API_CONTACT_EMAIL = ""

# API Configuration
app.config.REQUEST_TIMEOUT = 120
app.config.RESPONSE_TIMEOUT = 3600

# API V1 routes
app.blueprint(swagger_blueprint)
app.blueprint(v1)

@app.route('/')
async def get(request: Request) -> HTTPResponse:
    result = response.text('OK', 200)
    result.headers['content-type'] = 'text/plain'
    return result

def main():
    app.run(host="0.0.0.0", port=8080) #, single_process=True)

if __name__ == '__main__':
    main()
