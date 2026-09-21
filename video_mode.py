# Purpose: Process an uploaded video and record zone alert episodes.
from pathlib import Path
import csv
import io
import math
import tempfile

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO


# Draw the zone and check each person's bottom-center point.
def annotate_frame(frame, result, zone):
    zx1, zy1, zx2, zy2 = zone
    output = frame.copy()
    height, width = frame.shape[:2]
    cv2.rectangle(
        output, (zx1, zy1),
        (min(zx2, width - 1), min(zy2, height - 1)),
        (0, 255, 255), 2
    )
    inside_count = 0

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        foot_x, foot_y = (x1 + x2) / 2, y2
        inside = zx1 <= foot_x <= zx2 and zy1 <= foot_y <= zy2
        inside_count += int(inside)

        color = (0, 0, 255) if inside else (0, 200, 0)
        label = "INSIDE" if inside else "OUTSIDE"
        cv2.rectangle(
            output, (int(x1), int(y1)), (int(x2), int(y2)), color, 2
        )
        cv2.circle(output, (int(foot_x), int(foot_y)), 5, color, -1)
        cv2.putText(
            output, label, (int(x1), max(20, int(y1) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    return output, inside_count


# Show the video uploader, preview, processing, and report.
def run_video(model_path, confidence, horizontal, vertical):
    uploaded = st.file_uploader(
        "Upload a video", type=["mp4", "avi", "mov"], key="video_upload"
    )
    if uploaded is None:
        st.info("Upload your people-walking.mp4 video.")
        return

    seconds = st.slider("Process the first N seconds", 5, 30, 10)
    st.caption("Processing speed depends on your laptop CPU.")

    # OpenCV needs a file path. Use a separate temporary folder per run.
    with tempfile.TemporaryDirectory() as temporary_folder:
        video_path = Path(temporary_folder) / (
            "input" + Path(uploaded.name).suffix.lower()
        )
        video_path.write_bytes(uploaded.getvalue())
        capture = cv2.VideoCapture(str(video_path))

        try:
            ok, first_frame = capture.read()
            if not ok:
                st.error("Cannot read this video. Try a different MP4.")
                return

            fps = capture.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(fps) or fps <= 0:
                st.error("Cannot read the video's frame rate.")
                return

            # Calculate the zone using the sidebar percentages.
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
                preview, (zx1, zy1),
                (min(zx2, width - 1), min(zy2, height - 1)),
                (0, 255, 255), 3
            )
            st.image(preview, channels="BGR", caption="First frame: zone preview")

            if not st.button("Process Video"):
                return

            # Reuse the model within this browser session.
            if "person_model" not in st.session_state:
                st.session_state.person_model = YOLO(model_path)
            model = st.session_state.person_model

            # Confirm occupancy before alerting; wait before clearing.
            confirm_frames = max(1, math.ceil(fps * 0.2))
            clear_frames = max(1, math.ceil(fps * 0.5))
            positive_streak = 0
            empty_streak = 0
            alert_active = False
            events = []
            processed = 0
            max_inside = 0
            first_alert_image = None

            # Create display areas that update as frames are processed.
            frame_display = st.empty()
            status_display = st.empty()
            audio_display = st.empty()
            progress = st.progress(0.0)

            sample_rate = 22050
            t = np.arange(int(sample_rate * 0.3)) / sample_rate
            beep = 0.3 * np.sin(2 * np.pi * 880 * t)
            frame_limit = math.ceil(seconds * fps)

            for frame_index in range(frame_limit):
                if frame_index == 0:
                    frame = first_frame
                else:
                    ok, frame = capture.read()
                    if not ok:
                        break

                result = model.predict(
                    source=frame, device="cpu", imgsz=640,
                    conf=confidence, classes=[0], verbose=False
                )[0]
                annotated, inside_count = annotate_frame(frame, result, zone)
                max_inside = max(max_inside, inside_count)

                if inside_count > 0:
                    positive_streak += 1
                    empty_streak = 0
                else:
                    positive_streak = 0
                    empty_streak += 1

                # Create one event when occupancy becomes confirmed.
                if not alert_active and positive_streak >= confirm_frames:
                    alert_active = True
                    events.append({
                        "episode": len(events) + 1,
                        "confirmed_at_seconds": round(frame_index / fps, 2),
                        "persons_inside_at_confirmation": inside_count
                    })
                    audio_display.audio(
                        beep, sample_rate=sample_rate, autoplay=True
                    )
                    if first_alert_image is None:
                        first_alert_image = annotated.copy()

                # Allow another event after a sustained detection-free gap.
                if alert_active and empty_streak >= clear_frames:
                    alert_active = False
                    audio_display.empty()

                if alert_active:
                    status_display.error(
                        f"ALERT ACTIVE | Currently detected inside: {inside_count}"
                    )
                else:
                    status_display.info(
                        f"No confirmed alert | Currently detected inside: {inside_count}"
                    )

                processed += 1
                frame_display.image(
                    annotated, channels="BGR",
                    caption=f"Video time: {frame_index / fps:.2f} seconds"
                )
                progress.progress(min(processed / frame_limit, 1.0))

            # Summarize only the frames actually processed.
            progress.empty()
            st.write(f"Processed {processed} frames ({processed / fps:.2f} seconds).")
            st.metric("Alert episodes", len(events))
            st.metric("Maximum persons detected inside at once", max_inside)

            if events:
                st.dataframe(events, hide_index=True)

                # Export the episode report as CSV.
                report = io.StringIO()
                writer = csv.DictWriter(report, fieldnames=list(events[0]))
                writer.writeheader()
                writer.writerows(events)
                st.download_button(
                    "Download alert report",
                    report.getvalue(),
                    "video_alert_report.csv",
                    "text/csv",
                    on_click="ignore"
                )

                # Export a snapshot from the first confirmed alert.
                encoded_ok, encoded = cv2.imencode(".jpg", first_alert_image)
                if encoded_ok:
                    st.download_button(
                        "Download first alert snapshot",
                        encoded.tobytes(),
                        "video_alert_snapshot.jpg",
                        "image/jpeg",
                        on_click="ignore"
                    )
            else:
                st.info("No confirmed zone alert in the processed frames.")

        finally:
            # Release the video before deleting temporary files.
            capture.release()