# Copyright 2018 Airbus. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import base64
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import asyncio

from sanic import Blueprint, response
from sanic.request import Request
from sanic.response import HTTPResponse
from sanic.log import logger
from sanic_openapi import doc

from predict import Predict

v1 = Blueprint("v1", url_prefix="/api/v1")

predictor = Predict(logger=logger)

lock = None


@v1.listener("before_server_start")
async def init(sanic, loop):
    global lock
    lock = asyncio.Lock()


@v1.route("/openapi")
@doc.summary("Open API specification of this service in YAML format")
async def openapi(request: Request) -> HTTPResponse:
    return await response.file("tile_geo_process_api.yaml")


@v1.route("/health")
@doc.summary("Check if service is alive")
async def health(request: Request) -> HTTPResponse:
    result = response.text("OK", 200)
    result.headers["content-type"] = "text/plain"
    return result


@v1.route("/describe")
@doc.summary("The description of this service in JSON schema")
async def describe(request: Request) -> HTTPResponse:
    return await response.file("description.json")


@v1.route("/process", methods=["POST"])
@doc.summary("Launch processing service")
@doc.produces("JSON")
async def process_post(request: Request) -> HTTPResponse:
    data = request.json
    if not data:
        return response.text("JSON payload is empty, or payload is not JSON.", 500)

    if lock.locked():
        return response.text("A processing is already running.", 429)

    await lock.acquire()
    try:
        resolution = data["resolution"]
        tiles = [base64.b64decode(data["tiles"][i]) for i in range(len(data["tiles"]))]

        def _processing():
            return predictor.process(resolution, tiles)

        prediction = await asyncio.wait_for(async_exec(_processing), timeout=None)
        logging.debug("Prediction: %s", json.dumps(prediction))
        return response.json(prediction, 200)

    except Exception as e:
        err = {
            "message": "Error in inference.",
            "hint": str(e),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        logging.error(err)
        return response.json(err, 400)

    finally:
        lock.release()


async def async_exec(callback):
    with ThreadPoolExecutor(max_workers=1) as executor:
        return await asyncio.get_event_loop().run_in_executor(executor, callback)
