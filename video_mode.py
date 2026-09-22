# Purpose: Track people, detect zone entries, and repeat the alarm.
from pathlib import Path
import csv
import io
import math
import tempfile
import uuid
import wave

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO


# Create a short beep followed by exactly 2 seconds of silence.
def make_alarm():
    rate = 24000
    t = np.arange(int(rate * 0.25)) / rate
    tone = 0.3 * np.sin(2 * np.pi * 880 * t)
    sound = np.concatenate([tone, np.zeros(rate * 2)])

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes((sound * 32767).astype("<i2").tobytes())

    # Unique metadata makes each new entry restart the audio.
    data = (
        buffer.getvalue()
        + b"JUNK"
        + (16).to_bytes(4, "little")
        + uuid.uuid4().bytes
    )
    return (
        data[:4]
        + (len(data) - 8).to_bytes(4, "little")
        + data[8:]
    )


# Record outside-to-inside changes separately for each tracking ID.
def update_entries(states, observed, frame_index, missing_limit):
    for track_id in list(states):
        if frame_index - states[track_id]["last_seen"] > missing_limit:
            del states[track_id]

    new_events = []

    for track_id, inside in observed.items():
        previous = states.get(track_id)

        if inside and (previous is None or not previous["inside"]):
            kind = "first_seen_inside" if previous is None else "entry"
            new_events.append((track_id, kind))

        states[track_id] = {
            "inside": inside,
            "last_seen": frame_index
        }

    return new_events


# Draw the zone, tracking IDs, and each person's bottom-center point.
def annotate_frame(frame, result, zone):
    zx1, zy1, zx2, zy2 = zone
    output = frame.copy()
    height, width = frame.shape[:2]

    cv2.rectangle(
        output,
        (zx1, zy1),
        (min(zx2, width - 1), min(zy2, height - 1)),
        (0, 255, 255),
        2
    )

    observed = {}

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        foot_x, foot_y = (x1 + x2) / 2, y2

        inside = (
            zx1 <= foot_x <= zx2
            and zy1 <= foot_y <= zy2
        )

        if box.id is None:
            color = (0, 200, 255)
            label = "TRACK PENDING"
        else:
            track_id = int(box.id.item())
            observed[track_id] = inside
            color = (0, 0, 255) if inside else (0, 200, 0)
            label = f"ID {track_id}: {'INSIDE' if inside else 'OUTSIDE'}"

        cv2.rectangle(
            output,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            color,
            2
        )
        cv2.circle(
            output, (int(foot_x), int(foot_y)), 5, color, -1
        )
        cv2.putText(
            output,
            label,
            (int(x1), max(20, int(y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    return output, observed


def run_video(model_path, confidence, horizontal, vertical):
    uploaded = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov"],
        key="video_upload"
    )
    if uploaded is None:
        st.info("Upload a video to begin.")
        return

    seconds = st.slider("Process the first N seconds", 5, 30, 10)
    st.caption(
        "New zone entry: beep immediately after tracking detects it. "
        "While occupied: repeat with a 2-second silent gap."
    )

    with tempfile.TemporaryDirectory() as temporary_folder:
        video_path = Path(temporary_folder) / (
            "input" + Path(uploaded.name).suffix.lower()
        )
        video_path.write_bytes(uploaded.getvalue())
        capture = cv2.VideoCapture(str(video_path))
        audio_display = None

        try:
            ok, first_frame = capture.read()
            if not ok:
                st.error("Cannot read this video. Try another MP4.")
                return

            fps = capture.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or fps <= 0:
                st.error("Cannot read the video's frame rate.")
                return

            height, width = first_frame.shape[:2]
            zone = (
                int(horizontal[0] * width / 100),
                int(vertical[0] * height / 100),
                int(horizontal[1] * width / 100),
                int(vertical[1] * height / 100)
            )
            zx1, zy1, zx2, zy2 = zone

            if zx1 >= zx2 or zy1 >= zy2:
                st.warning("Select a zone with non-zero width and height.")
                return

            preview = first_frame.copy()
            cv2.rectangle(
                preview,
                (zx1, zy1),
                (min(zx2, width - 1), min(zy2, height - 1)),
                (0, 255, 255),
                3
            )
            st.image(
                preview, channels="BGR", caption="First frame: zone preview"
            )

            if not st.button("Process Video"):
                return

            # A fresh model gives each video run a fresh tracker.
            model = YOLO(model_path)

            states = {}
            events = []
            processed = 0
            max_inside = 0
            first_alert_image = None
            alarm_playing = False

            # Remember an ID through brief missed detections.
            missing_limit = max(1, math.ceil(fps * 0.5))
            frame_limit = math.ceil(seconds * fps)

            frame_display = st.empty()
            status_display = st.empty()
            audio_display = st.empty()
            progress = st.progress(0.0)

            for frame_index in range(frame_limit):
                if frame_index == 0:
                    frame = first_frame
                else:
                    ok, frame = capture.read()
                    if not ok:
                        break

                # Track the same people across consecutive frames.
                result = model.track(
                    source=frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    device="cpu",
                    imgsz=640,
                    conf=confidence,
                    classes=[0],
                    verbose=False
                )[0]

                annotated, observed = annotate_frame(
                    frame, result, zone
                )

                inside_ids = sorted(
                    track_id
                    for track_id, inside in observed.items()
                    if inside
                )
                inside_count = len(inside_ids)
                max_inside = max(max_inside, inside_count)

                new_events = update_entries(
                    states, observed, frame_index, missing_limit
                )

                for track_id, kind in new_events:
                    events.append({
                        "alert_no": len(events) + 1,
                        "track_id": track_id,
                        "event": kind,
                        "video_time_seconds": round(frame_index / fps, 2)
                    })

                if new_events and first_alert_image is None:
                    first_alert_image = annotated.copy()

                # Restart immediately for a new entry.
                # Otherwise the browser repeats the beep + silent gap.
                if inside_count > 0:
                    if new_events or not alarm_playing:
                        audio_display.audio(
                            make_alarm(),
                            format="audio/wav",
                            autoplay=True,
                            loop=True
                        )
                        alarm_playing = True

                    status_display.error(
                        f"INTRUSION ALERT | Inside: {inside_count} "
                        f"| Tracking IDs: {inside_ids}"
                    )
                else:
                    if alarm_playing:
                        audio_display.empty()
                        alarm_playing = False

                    status_display.success(
                        "No tracked person detected inside the zone."
                    )

                processed += 1
                frame_display.image(
                    annotated,
                    channels="BGR",
                    caption=f"Video time: {frame_index / fps:.2f} seconds"
                )
                progress.progress(min(processed / frame_limit, 1.0))

            # Stop the alarm when this processing run ends.
            audio_display.empty()
            progress.empty()
            status_display.info("Processing finished. Alarm stopped.")

            st.write(
                f"Processed {processed} frames "
                f"({processed / fps:.2f} seconds)."
            )
            st.metric("Entry / first-seen-inside alerts", len(events))
            st.metric("Maximum tracked persons inside at once", max_inside)
            st.caption(
                "first_seen_inside: an ID first appeared inside the zone. "
                "entry: an observed ID moved from outside to inside. "
                "Re-entry can create another alert for the same ID."
            )

            if events:
                st.dataframe(events, hide_index=True)

                report = io.StringIO()
                writer = csv.DictWriter(
                    report, fieldnames=list(events[0])
                )
                writer.writeheader()
                writer.writerows(events)

                st.download_button(
                    "Download alert report",
                    report.getvalue(),
                    "video_alert_report.csv",
                    "text/csv",
                    on_click="ignore"
                )

                encoded_ok, encoded = cv2.imencode(
                    ".jpg", first_alert_image
                )
                if encoded_ok:
                    st.download_button(
                        "Download first alert snapshot",
                        encoded.tobytes(),
                        "video_alert_snapshot.jpg",
                        "image/jpeg",
                        on_click="ignore"
                    )
            else:
                st.info("No tracked zone entries in the processed frames.")

        finally:
            capture.release()
            if audio_display is not None:
                audio_display.empty()