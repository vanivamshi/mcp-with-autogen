"""Flask REST API that routes natural-language requests to the AutoGen agent."""

import asyncio
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from agent import run_task

load_dotenv()

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/run")
def run():
    data = request.get_json(silent=True) or {}
    task = data.get("task", "").strip()
    if not task:
        return jsonify({"status": "error", "message": "Missing 'task' in JSON body"}), 400

    try:
        result = asyncio.run(run_task(task))
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7001"))

    if os.getenv("NGROK_AUTHTOKEN"):
        from pyngrok import ngrok
        from pyngrok.exception import PyngrokError

        try:
            ngrok.set_auth_token(os.environ["NGROK_AUTHTOKEN"])
            public_url = ngrok.connect(port).public_url
            print(f" * ngrok public URL: {public_url}  ->  POST {public_url}/run")
        except PyngrokError as e:
            print(f" * ngrok failed, serving locally only: {str(e).splitlines()[0][:200]}")

    app.run(host="0.0.0.0", port=port)
