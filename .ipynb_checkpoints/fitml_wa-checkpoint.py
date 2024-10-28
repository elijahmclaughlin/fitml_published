import os
import tensorflow as tf
import numpy as np
import imageio
from PIL import Image, ImageDraw
import streamlit as st
from tempfile import NamedTemporaryFile

# MoveNet Lightning model
def load_movenet_model():
    interpreter = tf.lite.Interpreter(model_path="movenet.tflite")
    interpreter.allocate_tensors()
    return interpreter

movenet_interpreter = load_movenet_model()

def movenet_predict(image, interpreter):
    """Runs pose estimation using a TensorFlow Lite model."""
    input_image = tf.image.resize_with_pad(image, 192, 192)
    input_image = tf.cast(input_image, dtype=tf.uint8)
    input_image = np.expand_dims(input_image, axis=0)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]['index']).reshape((17, 3))
    return keypoints

def calculate_angle(a, b, c):
    """Calculates the angle between three points for pose landmarks."""
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

def analyze_squat(keypoints):
    """Calculates squat depth based on hip, knee, and ankle."""
    hip, knee, ankle = keypoints[11], keypoints[13], keypoints[15]
    angle = calculate_angle(hip, knee, ankle)
    squat_depth = "Deep Squat" if angle < 90 else "Normal Squat" if angle < 120 else "Shallow Squat"
    return angle, squat_depth

# left side keypoints
def analyze_bench(keypoints):
    """Calculates bench press depth based on shoulder, elbow, and wrist."""
    shoulder, elbow, wrist = keypoints[5], keypoints[7], keypoints[9]
    angle = calculate_angle(shoulder, elbow, wrist)
    bench_depth = "Full Press" if angle < 90 else "Partial Press" if angle < 150 else "No Full Extension"
    return angle, bench_depth

def analyze_deadlift(keypoints):
    """Calculates deadlift form based on hip, knee, and ankle."""
    hip, knee, ankle = keypoints[11], keypoints[13], keypoints[15]
    angle = calculate_angle(hip, knee, ankle)
    deadlift_depth = "Proper Form" if angle < 80 else "Slight Bend" if angle < 120 else "High Bend"
    return angle, deadlift_depth

# Streamlit UI
st.title("Exercise Analysis with FitML")
exercise_type = st.selectbox("Select Exercise", ("Squat", "Bench Press", "Deadlift"))
uploaded_file = st.file_uploader("Upload your exercise video", type=["mp4", "mov"])

# analyzing uploaded video
if uploaded_file:
    original_filename = uploaded_file.name
    base_name, ext = os.path.splitext(original_filename)
    output_filename = f"{base_name}_analyzed_fitml{ext}"
    
    with NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name

    output_path = NamedTemporaryFile(suffix=ext, delete=False).name

    reader = imageio.get_reader(video_path, 'ffmpeg')
    fps = reader.get_meta_data()['fps']
    writer = imageio.get_writer(output_path, fps=fps)

    # Add progress bar
    progress_bar = st.progress(0)
    total_frames = reader.count_frames()
    frame_count = 0

    # edit video
    for frame in reader:
        frame_rgb = Image.fromarray(frame).convert("RGB")
        frame_np = np.array(frame_rgb)

        keypoints = movenet_predict(frame_np)

        if exercise_type == "Squat":
            angle, depth = analyze_squat(keypoints)
        elif exercise_type == "Bench Press":
            angle, depth = analyze_bench(keypoints)
        elif exercise_type == "Deadlift":
            angle, depth = analyze_deadlift(keypoints)

        draw = ImageDraw.Draw(frame_rgb)
        draw.text((50, 50), f'Angle: {int(angle)}', fill=(255, 255, 255))
        draw.text((50, 100), depth, fill=(0, 255, 0))

        for i, (y, x, c) in enumerate(keypoints):
            if c > 0.5:
                draw.ellipse((x * frame_rgb.width - 5, y * frame_rgb.height - 5,
                              x * frame_rgb.width + 5, y * frame_rgb.height + 5),
                             fill=(0, 255, 0))

        frame_with_overlays = np.array(frame_rgb)
        writer.append_data(frame_with_overlays)
        
        # progress bar
        frame_count += 1
        progress_bar.progress(frame_count / total_frames)

    reader.close()
    writer.close()

    # display video
    st.video(output_path)

    # download button
    with open(output_path, "rb") as file:
        btn = st.download_button(
            label="Download Analyzed Video",
            data=file,
            file_name=output_filename,
            mime="video/mp4"
        )
