# Minimal Flask Inference Service

This directory demonstrates how to wrap the YOLOv5 lung detector inside a lightweight REST endpoint so that other components in the pipeline can request predictions over HTTP.

## Prerequisites
- Python 3.8 or newer.
- Install Flask and Requests inside your active environment:
  ```bash
  pip install Flask requests
  ```

## Launching the API
1. Activate the environment that already contains the project dependencies.
2. Start the server with a custom port if desired:
   ```bash
   python3 restapi.py --port 5000
   ```
3. The service exposes the prediction route at `/v1/object-detection/yolov5s`.

## Example Request
Send any RGB image using `curl`:
```bash
curl -X POST -F image=@sample.jpg "http://localhost:5000/v1/object-detection/yolov5s"
```
The response returns a JSON array of bounding boxes with class indices, labels, confidence values, and normalized coordinates.

For programmatic use, check `example_request.py`, which streams the file, posts it to the endpoint, and prints the parsed detections.

## Troubleshooting
- Ensure the YOLO weights referenced by `restapi.py` are downloaded before launching.
- Warming up the model after startup can reduce latency; send a single request with a small image to trigger initialization.
- Use environment variables like `FLASK_ENV=development` for verbose logging during debugging.
