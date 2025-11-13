# REST Endpoint for Lung Detection

Use this sample service when you need to expose the lung-focused YOLOv5 checkpoints through a simple HTTP interface for integration tests or rapid demos.

## Getting Ready
- Confirm your Python environment already contains the main repository dependencies.
- Add the lightweight web stack:
  ```bash
  pip install Flask requests
  ```

## Starting the Server
1. Move into this folder.
2. Run the launcher with any supported port:
   ```bash
   python3 restapi.py --port 5000
   ```
3. The application mounts a single POST route at `/v1/object-detection/yolov5s`.

## Calling the API
```bash
curl -X POST -F image=@chest.png "http://localhost:5000/v1/object-detection/yolov5s"
```
Returned payload fields:
- `class`: numeric index from the model metadata.
- `name`: readable category label.
- `confidence`: probability score between 0 and 1.
- `xmin`, `ymin`, `xmax`, `ymax`: pixel coordinates in the original image space.

Refer to `example_request.py` for a minimal Python client that prepares the POST request and handles JSON decoding.

## Tips
- Load the detector weights ahead of time; the script expects them in the relative paths defined inside `restapi.py`.
- You can adjust batch size and confidence thresholds in the handler to align with production requirements.
- Pair this API with a reverse proxy like Nginx when deploying to shared environments.

