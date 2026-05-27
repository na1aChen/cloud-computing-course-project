import os
from flask import Flask, jsonify
import redis

app = Flask(__name__)

redis_client = redis.Redis(
    host=os.environ.get("REDIS_HOST", "localhost"),
    port=int(os.environ.get("REDIS_PORT", 6379)),
    password=os.environ.get("REDIS_PASSWORD", "") or None,
    decode_responses=True,
    socket_connect_timeout=2,
)


@app.route("/api/ping")
def ping():
    return jsonify({"status": "ok"})


@app.route("/api/count")
def count():
    try:
        visits = redis_client.incr("visit_count")
    except redis.RedisError:
        visits = -1
    return jsonify({"visits": visits})


@app.route("/api/health")
def health():
    try:
        redis_client.ping()
        redis_status = "connected"
    except redis.RedisError:
        redis_status = "disconnected"
    return jsonify({"redis": redis_status})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
