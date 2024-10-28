import os
import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
from PIL import Image, ImageDraw
import streamlit as st
from tempfile import NamedTemporaryFile

# TFLite model
model_path = "/mount/src/fitml_published/movenet.tflite"
st.write(f"Attempting to load model from: {model_path}")

def load_model(model_path):
    """Attempts to load the TensorFlow Lite model and returns the interpreter if successful."""
    st.write(f"Attempting to load model from: {model_path}")
    try:
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        st.success("Model loaded successfully!")
        return interpreter
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please verify the path.")
    except Exception as e:
        st.error(f"Failed to load the model: {e}")
    return None

interpreter = load_model(model_path)

def movenet_predict(image, interpreter):
    """Runs pose estimation using the TensorFlow Lite model."""
    input_image = np.array(image.resize((192, 192)))
    input_image = np.expand_dims(input_image, axis=0)
    input_image = np.array(input_image, dtype=np.uint8)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]['index']).reshape((17, 3))
    return keypoints

def calculate_angle(a, b, c):
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

def analyze_squat(keypoints):
    hip, knee, ankle = keypoints[11], keypoints[13], keypoints[15]
    angle = calculate_angle(hip, knee, ankle)
    squat_depth = "Deep Squat" if angle < 90 else "Normal Squat" if angle < 120 else "Shallow Squat"
    return angle, squat_depth

def analyze_bench(keypoints):
    shoulder, elbow, wrist = keypoints[5], keypoints[7], keypoints[9]
    angle = calculate_angle(shoulder, elbow, wrist)
    bench_depth = "Full Press" if angle < 90 else "Partial Press" if angle < 150 else "No Full Extension"
    return angle, bench_depth

def analyze_deadlift(keypoints):
    hip, knee, ankle = keypoints[11], keypoints[13], keypoints[15]
    angle = calculate_angle(hip, knee, ankle)
    deadlift_depth = "Proper Form" if angle < 80 else "Slight Bend" if angle < 120 else "High Bend"
    return angle, deadlift_depth

# Keypoint connections
connections = [
    (5, 7), (7, 9),   # Left arm
    (6, 8), (8, 10),  # Right arm
    (5, 11), (11, 13), (13, 15),  # Left side
    (6, 12), (12, 14), (14, 16),  # Right side
    (5, 6), (11, 12)  # Shoulders and hips
]

st.title("Exercise Analysis with FitML")
exercise_type = st.selectbox("Select Exercise", ("Squat", "Bench Press", "Deadlift"))
uploaded_file = st.file_uploader("Upload your exercise video", type=["mp4", "mov"])

if uploaded_file:
    original_filename = uploaded_file.name
    base_name, ext = os.path.splitext(original_filename)
    output_filename = f"{base_name}_analyzed_fitml{ext}"
    
    with NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input_file:
        temp_input_file.write(uploaded_file.read())
        input_video_path = temp_input_file.name

    output_video_path = NamedTemporaryFile(delete=False, suffix=".mp4").name

    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    frame_count = 0

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        keypoints = movenet_predict(frame_rgb, interpreter)

        if exercise_type == "Squat":
            angle, depth = analyze_squat(keypoints)
        elif exercise_type == "Bench Press":
            angle, depth = analyze_bench(keypoints)
        elif exercise_type == "Deadlift":
            angle, depth = analyze_deadlift(keypoints)

        draw = ImageDraw.Draw(frame_rgb)
        for start, end in connections:
            if keypoints[start][2] > 0.5 and keypoints[end][2] > 0.5:
                y1, x1 = keypoints[start][:2]
                y2, x2 = keypoints[end][:2]
                draw.line(
                    [(x1 * frame_rgb.width, y1 * frame_rgb.height), (x2 * frame_rgb.width, y2 * frame_rgb.height)],
                    fill=(255, 255, 255),
                    width=3
                )

        frame_processed = cv2.cvtColor(np.array(frame_rgb), cv2.COLOR_RGB2BGR)

        text_position = (50, 50)
        cv2.putText(frame_processed, f'Angle: {int(angle)}', (text_position[0], text_position[1]),
                    font, font_scale, (0, 0, 0), font_thickness + 2, lineType=cv2.LINE_AA)
        cv2.putText(frame_processed, f'Angle: {int(angle)}', (text_position[0], text_position[1]),
                    font, font_scale, (255, 255, 255), font_thickness, lineType=cv2.LINE_AA)

        text_position_depth = (50, 100)
        cv2.putText(frame_processed, depth, (text_position_depth[0], text_position_depth[1]),
                    font, font_scale, (0, 0, 0), font_thickness + 2, lineType=cv2.LINE_AA)
        cv2.putText(frame_processed, depth, (text_position_depth[0], text_position_depth[1]),
                    font, font_scale, (0, 255, 0), font_thickness, lineType=cv2.LINE_AA)

        for i, (y, x, c) in enumerate(keypoints):
            if c > 0.5:
                cv2.circle(frame_processed, (int(x * width), int(y * height)), 5, (0, 255, 0), -1)

        out.write(frame_processed)
        
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)

    cap.release()
    out.release()

    with open(output_video_path, "rb") as file:
        st.download_button(
            label="Download Analyzed Video",
            data=file,
            file_name=output_filename,
            mime="video/mp4"
        )
