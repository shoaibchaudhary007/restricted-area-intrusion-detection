# Restricted Area Intrusion Detection & Alert System

A Python application that detects people in uploaded images and videos, checks whether they are inside a selected restricted zone, and generates visual and audible alerts.

Built as an AI Automation course project using a fine-tuned YOLO11n model and Streamlit.

## Features

- Person detection in images and videos.
- Adjustable rectangular restricted zone.
- Adjustable detection confidence threshold.
- Red boxes for people inside the zone and green boxes for people outside.
- ByteTrack tracking IDs in video mode.
- An audible alert when a tracked person enters the zone.
- Repeating alarm with a 2-second silent gap while tracked people remain inside.
- Alarm stops when no tracked person is detected inside or processing ends.
- Downloadable image results, video alert CSV reports, and first-alert snapshots.

## How Zone Detection Works

The application uses the bottom-center point of each person's bounding box as an estimate of their position on the ground.

If this point is inside the selected rectangle, the person is marked INSIDE. Otherwise, they are marked OUTSIDE.

A person's upper body may overlap the zone while their bottom-center point remains outside.

## Model and Training

| Setting | Value |
|---|---|
| Model | YOLO11n |
| Starting weights | Pretrained `yolo11n.pt` |
| Method | Fine-tuning |
| Target class | `person` |
| Class ID | 0 |
| Training epochs | 50 |
| Image size | 640 |
| Batch size | 16 |
| Training hardware | Kaggle Tesla T4 GPU |
| Saved model | `best.pt` |

The model was fine-tuned from pretrained weights, rather than trained with randomly initialized weights.

## Dataset

Source: [Person dataset by Samuel Ayman on Kaggle](https://www.kaggle.com/datasets/samuelayman/person)

| Split | Images | Annotated person boxes |
|---|---:|---:|
| Training | 749 | 2,685 |
| Validation | 200 | 769 |
| Test | 50 | 220 |
| Total | 999 | 3,674 |

Labels use YOLO bounding-box format:

`class_id center_x center_y width height`

Coordinates are normalized relative to image dimensions. Label checks found no missing label files, empty label files, or label-format errors.

## Test Results

The saved model was evaluated on the dataset's test split of 50 images containing 220 annotated person boxes.

| Metric | Result |
|---|---:|
| Precision | 77.71% |
| Recall | 79.55% |
| mAP@50 | 85.35% |
| mAP@50–95 | 61.76% |

These metrics evaluate person detection. They do not measure tracking accuracy or the accuracy of the complete intrusion alert system.

## Technologies

| Technology | Purpose |
|---|---|
| Python | Application logic |
| Ultralytics YOLO | Person detection and tracking interface |
| PyTorch | Model execution |
| ByteTrack | Tracking people across video frames |
| OpenCV | Reading video frames and drawing annotations |
| Streamlit | Web interface |
| NumPy | Image arrays and alarm waveform generation |
| Pillow | Image loading and orientation handling |
| lap | Assignment operations used by tracking |

## Repository Files

| File | Purpose |
|---|---|
| `app.py` | Main Streamlit application and image mode |
| `video_mode.py` | Video processing, tracking, alarms, and reports |
| `best.pt` | Fine-tuned person detection model |
| `requirements.txt` | Python dependencies |
| `packages.txt` | System packages for cloud deployment |
| `README.md` | Project documentation |

## Run Locally

Use Python 3.11. Download or clone this repository and open a terminal in its folder.

Create and activate a Conda environment:

```bash
conda create -n intrusion python=3.11 -y
conda activate intrusion
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the application:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, normally:

http://localhost:8501

Keep `best.pt` in the same folder as `app.py`.

## Using the Application

1. Choose Image or Video mode.
2. Upload a supported file.
3. Adjust the confidence threshold.
4. Set the horizontal and vertical zone boundaries.
5. Click Check Restricted Area or Process Video.
6. Review the detections and alerts.
7. Download the available results.

Video mode processes a selected initial segment of the uploaded video.

## Video Alert Report

The CSV report contains:

- `alert_no`: Sequential event number.
- `track_id`: Tracking ID associated with the event.
- `event`: `first_seen_inside` or `entry`.
- `video_time_seconds`: Event timestamp in the source video.

`first_seen_inside` means the tracker first observed that ID inside the zone.

`entry` means an observed ID changed from outside to inside. Re-entry can produce another event for the same ID.

Repeated reminder beeps do not create additional CSV rows. Tracking IDs are temporary labels, not personal identities.

## Deployment

The application is deployed on Streamlit Community Cloud using this GitHub repository.

Deployment configuration:

- Branch: `main`
- Main file: `app.py`
- Python version: `3.11`

## Limitations

- CPU processing may run slower than the source video's playback speed.
- Small, overlapping, or partially visible people may be missed.
- Tracking IDs can change, which may produce additional entry alerts.
- The bottom-center point is an approximation of ground position.
- Audio playback depends on browser autoplay permissions.
- This is an educational prototype and has not been validated as a production security system.

## Credits

- Person dataset: Samuel Ayman, Kaggle.
- Detection framework: Ultralytics YOLO.
- Tracking method: ByteTrack.
- Web interface: Streamlit.
- Demonstration video: [Roboflow sample video](https://media.roboflow.com/supervision/video-examples/people-walking.mp4).
