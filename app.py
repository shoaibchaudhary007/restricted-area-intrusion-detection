# Purpose: Detect people inside a user-selected restricted zone.
from pathlib import Path
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO

# Configure the page.
st.set_page_config(page_title="Intrusion Detection", layout="wide")
st.title("Restricted Area Intrusion Detection & Alert System")
st.caption("Select a restricted area and check for people inside it.")

# Locate the saved model.
model_path = Path(__file__).resolve().parent / "best.pt"
if not model_path.is_file():
    st.error("Place best.pt in the same folder as app.py.")
    st.stop()

# Choose detection confidence and zone boundaries.
st.sidebar.header("Detection settings")
confidence = st.sidebar.slider(
    "Minimum confidence", 0.10, 0.90, 0.25, 0.05
)
st.sidebar.header("Restricted zone")
horizontal = st.sidebar.slider(
    "Left to right (%)", 0, 100, (0, 50)
)
vertical = st.sidebar.slider(
    "Top to bottom (%)", 0, 100, (60, 100)
)
# Choose between the image and video features.
mode = st.sidebar.radio("Input type", ["Image", "Video"])

if mode == "Video":
    from video_mode import run_video
    run_video(model_path, confidence, horizontal, vertical)
    st.stop()
uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"]
)
if uploaded_file is None:
    st.info("Upload an image to begin.")
    st.stop()

# Read the image and correct its orientation.
try:
    image = ImageOps.exif_transpose(
        Image.open(uploaded_file)
    ).convert("RGB")
except (OSError, ValueError):
    st.error("Cannot read this image. Upload a valid JPG or PNG.")
    st.stop()

# Convert percentages into image coordinates.
width, height = image.size
zx1 = int(horizontal[0] * width / 100)
zx2 = int(horizontal[1] * width / 100)
zy1 = int(vertical[0] * height / 100)
zy2 = int(vertical[1] * height / 100)

if zx1 >= zx2 or zy1 >= zy2:
    st.warning("Select a zone with non-zero width and height.")
    st.stop()

# Prepare an OpenCV image and draw the yellow zone.
frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
preview = frame.copy()
cv2.rectangle(
    preview,
    (zx1, zy1),
    (min(zx2, width - 1), min(zy2, height - 1)),
    (0, 255, 255),
    3
)

left_column, right_column = st.columns(2)
left_column.image(
    preview, channels="BGR", caption="Yellow rectangle: restricted zone"
)

if st.button("Check Restricted Area"):
    # Keep a separate model for this browser session.
    with st.spinner("Checking for people..."):
        if "person_model" not in st.session_state:
            st.session_state.person_model = YOLO(model_path)

        result = st.session_state.person_model.predict(
            source=frame,
            device="cpu",
            imgsz=640,
            conf=confidence,
            classes=[0],
            verbose=False
        )[0]

    annotated = preview.copy()
    inside_count = 0

    # Check each detected person's bottom-center point.
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        foot_x = (x1 + x2) / 2
        foot_y = y2
        inside = zx1 <= foot_x <= zx2 and zy1 <= foot_y <= zy2

        if inside:
            inside_count += 1

        # Red means inside; green means outside.
        color = (0, 0, 255) if inside else (0, 200, 0)
        label = "INSIDE" if inside else "OUTSIDE"
        score = float(box.conf[0].item())

        cv2.rectangle(
            annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2
        )
        cv2.circle(
            annotated, (int(foot_x), int(foot_y)), 6, color, -1
        )
        cv2.putText(
            annotated,
            f"{label} {score:.2f}",
            (int(x1), max(20, int(y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # Display the result and visual alert.
    right_column.image(
        annotated, channels="BGR", caption="Zone detection result"
    )
    total_column, inside_column = st.columns(2)
    total_column.metric("Persons detected", len(result.boxes))
    inside_column.metric("Persons inside zone", inside_count)

       # Play two short beeps when a person is inside the zone.
    if inside_count > 0:
        st.error(f"INTRUSION ALERT: {inside_count} person(s) inside the zone!")

        sample_rate = 22050
        duration = 0.25
        time_points = np.arange(int(sample_rate * duration)) / sample_rate

        beep = 0.3 * np.sin(2 * np.pi * 880 * time_points)
        silence = np.zeros(int(sample_rate * 0.15))
        alarm = np.concatenate([beep, silence, beep])

        st.audio(alarm, sample_rate=sample_rate, autoplay=True)
        st.caption("If the alarm does not play automatically, press Play.")
    else:
        st.success("No person detected inside the selected zone.")

    # Add detection counts to the evidence image.
    evidence = annotated.copy()
    status = f"Detected: {len(result.boxes)} | Inside zone: {inside_count}"
    cv2.rectangle(evidence, (0, 0), (width - 1, 45), (0, 0, 0), -1)
    cv2.putText(
        evidence, status, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
    )

    # Let the user download the result as a JPEG image.
    encoded_ok, encoded_image = cv2.imencode(".jpg", evidence)
    if encoded_ok:
        filename = "intrusion_alert.jpg" if inside_count > 0 else "zone_clear.jpg"
        st.download_button(
            "Download result image",
            data=encoded_image.tobytes(),
            file_name=filename,
            mime="image/jpeg",
            on_click="ignore"
        )
    else:
        st.warning("Could not prepare the result image for download.")