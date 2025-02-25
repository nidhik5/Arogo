"""code captures selective facial features to make predictions not normalised and uses a mistral model  output is only emotion probabilities"""

import cv2
import mediapipe as mp
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Mistral-7B Model Locally
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Function to extract and normalize facial features
def extract_facial_features(landmarks):
    # Define key facial landmark indices
    LEFT_EYEBROW = [55, 65, 52]
    RIGHT_EYEBROW = [285, 295, 282]
    LEFT_EYE = [159, 145]
    RIGHT_EYE = [386, 374]
    MOUTH = [13, 14]  # Upper & lower lip
    SMILE_WIDTH = [61, 291]  # Left & right mouth corners

    # Convert landmarks to NumPy array
    landmark_array = np.array([(lm.x, lm.y) for lm in landmarks])

    # Compute reference distance for normalization (Interocular Distance)
    interocular_distance = np.linalg.norm(landmark_array[LEFT_EYE[0]] - landmark_array[RIGHT_EYE[0]])

    # Compute features and normalize
    eyebrow_raise = (np.mean(landmark_array[LEFT_EYEBROW, 1]) - np.mean(landmark_array[LEFT_EYE, 1])) / interocular_distance
    mouth_opening = np.linalg.norm(landmark_array[MOUTH[0]] - landmark_array[MOUTH[1]]) / interocular_distance
    eye_openness = np.linalg.norm(landmark_array[LEFT_EYE[0]] - landmark_array[LEFT_EYE[1]]) / interocular_distance
    smile_width = np.linalg.norm(landmark_array[SMILE_WIDTH[0]] - landmark_array[SMILE_WIDTH[1]]) / interocular_distance  # ✅ New Feature

    return {
        "eyebrow_raise": eyebrow_raise,
        "mouth_opening": mouth_opening,
        "eye_openness": eye_openness,
        "smile_width": smile_width  # ✅ Included in output
    }

# Function to generate emotion description using Mistral-7B
def generate_emotion_description(features):
    prompt = f"""
    The person has these facial expressions:
    - Eyebrow Raise: {features['eyebrow_raise']:.3f}
    - Mouth Opening: {features['mouth_opening']:.3f}
    - Eye Openness: {features['eye_openness']:.3f}
    - Smile Width: {features['smile_width']:.3f}

    Describe their likely emotion in a detailed and human-like manner.
    """

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_length=100)
    description = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return description

# Open Webcam
cap = cv2.VideoCapture(0)

# Ensure webcam opens correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break


    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            features = extract_facial_features(face_landmarks.landmark)
            emotion_text = generate_emotion_description(features)

            # Display the output on screen
            y0, dy = 30, 25
            for i, line in enumerate(emotion_text.split(". ")):
                y = y0 + i * dy
                cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("Emotion Detection", frame)

    # Quit on 'q' press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
