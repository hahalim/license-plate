import streamlit as st
import cv2
import numpy as np
import easyocr
import tempfile
import os

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

def find_license_plate(image):
    edges = preprocess_image(image)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    license_plate_contour = None
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        if 2 < aspect_ratio < 5 and 1000 < cv2.contourArea(contour) < 50000:
            license_plate_contour = contour
            break
    
    if license_plate_contour is not None:
        x, y, w, h = cv2.boundingRect(license_plate_contour)
        return (x, y, w, h)
    return None

def iou(box1, box2):
    # Calculate intersection over union
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def process_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    video = cv2.VideoCapture(tfile.name)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
    
    reader = easyocr.Reader(['en'])
    detected_plates = []

    current_plate_bbox = None
    frames_since_detection = 0
    max_frames_without_detection = 30  # Adjust as needed

    for frame_count in range(total_frames):
        ret, frame = video.read()
        if not ret:
            break

        draw_frame = frame.copy()
        plate_bbox = find_license_plate(frame)

        if plate_bbox:
            if current_plate_bbox is None or iou(plate_bbox, current_plate_bbox) < 0.5:
                current_plate_bbox = plate_bbox
            frames_since_detection = 0
        else:
            frames_since_detection += 1

        if frames_since_detection > max_frames_without_detection:
            current_plate_bbox = None

        if current_plate_bbox:
            x, y, w, h = current_plate_bbox
            plate_img = frame[y:y+h, x:x+w]

            results = reader.readtext(plate_img)

            for (_, text, prob) in results:
                if prob > 0.5:
                    detected_plates.append(text)
                    cv2.rectangle(draw_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(draw_frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        out.write(draw_frame)

        if frame_count % 30 == 0:
            progress = (frame_count + 1) / total_frames
            st.progress(progress)

    video.release()
    out.release()
    return list(set(detected_plates))

def process_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    video = cv2.VideoCapture(tfile.name)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = 'output.mp4'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    reader = easyocr.Reader(['en'])
    detected_plates = []

    current_plate_bbox = None
    frames_since_detection = 0
    max_frames_without_detection = 30  # Adjust as needed

    for frame_count in range(total_frames):
        ret, frame = video.read()
        if not ret:
            break

        draw_frame = frame.copy()
        plate_bbox = find_license_plate(frame)

        if plate_bbox:
            if current_plate_bbox is None or iou(plate_bbox, current_plate_bbox) < 0.5:
                current_plate_bbox = plate_bbox
            frames_since_detection = 0
        else:
            frames_since_detection += 1

        if frames_since_detection > max_frames_without_detection:
            current_plate_bbox = None

        if current_plate_bbox:
            x, y, w, h = current_plate_bbox
            plate_img = frame[y:y+h, x:x+w]

            results = reader.readtext(plate_img)

            for (_, text, prob) in results:
                if prob > 0.5:
                    detected_plates.append(text)
                    cv2.rectangle(draw_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(draw_frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        out.write(draw_frame)

        if frame_count % 30 == 0:
            progress = (frame_count + 1) / total_frames
            st.progress(progress)

    video.release()
    out.release()
    return list(set(detected_plates)), output_path

def main():
    st.title("License Plate Extractor")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        st.video(uploaded_file)

        if 'processed_video_path' not in st.session_state:
            st.session_state.processed_video_path = None

        if st.button("Extract License Plates"):
            with st.spinner("Processing video..."):
                plates, output_path = process_video(uploaded_file)
                st.session_state.processed_video_path = output_path

            st.success("License plate extraction complete!")
            
            st.video(output_path)

        # Display download button if processed video exists
        if st.session_state.processed_video_path:
            with open(st.session_state.processed_video_path, 'rb') as file:
                st.download_button(
                    label="Download processed video",
                    data=file,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )

if __name__ == "__main__":
    main()
