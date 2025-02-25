"""code with normalised input features only return objective 6 metrics"""

import cv2
import mediapipe as mp
import numpy as np
import time
from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch

# Load a pre-trained Vision Transformer (ViT) model for emotion recognition
model_name = "google/vit-base-patch16-224"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

# Load the model with 22 output classes and ignore size mismatch
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=22,  # Set the number of output classes to 22
    ignore_mismatched_sizes=True  # Ignore size mismatch in the output layer
)

# Emotion labels for 22 classes (example)
emotion_labels = [
    "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral",
    "Contempt", "Anxiety", "Calm", "Confusion", "Desire", "Empathy",
    "Excitement", "Interest", "Pride", "Relief", "Shame", "Satisfaction",
    "Boredom", "Amusement", "Awe"
]

# Polarity values for emotions (example)
emotion_polarity = {
    "Angry": -1.0, "Disgust": -0.9, "Fear": -0.8, "Happy": 1.0, "Sad": -1.0,
    "Surprise": 0.5, "Neutral": 0.0, "Contempt": -0.7, "Anxiety": -0.6,
    "Calm": 0.8, "Confusion": -0.5, "Desire": 0.7, "Empathy": 0.6,
    "Excitement": 0.9, "Interest": 0.5, "Pride": 0.8, "Relief": 0.7,
    "Shame": -0.9, "Satisfaction": 0.8, "Boredom": -0.4, "Amusement": 0.7, "Awe": 0.6
}

# Arousal levels for emotions (example)
emotion_arousal = {
    "Angry": "High", "Disgust": "Low", "Fear": "High", "Happy": "Medium", "Sad": "Low",
    "Surprise": "High", "Neutral": "Low", "Contempt": "Medium", "Anxiety": "High",
    "Calm": "Low", "Confusion": "Medium", "Desire": "Medium", "Empathy": "Medium",
    "Excitement": "High", "Interest": "Medium", "Pride": "Medium", "Relief": "Medium",
    "Shame": "Low", "Satisfaction": "Medium", "Boredom": "Low", "Amusement": "Medium", "Awe": "Medium"
}

# Initialize MediaPipe FaceMesh for facial landmark detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

# Variables for storing emotion data
emotion_history = []
start_time = time.time()
last_processed_time = start_time

# Function to normalize facial features by face size
def normalize_features(landmarks, face_width):
    # Extract key landmarks
    left_eye = np.array([landmarks[33].x, landmarks[33].y])
    right_eye = np.array([landmarks[263].x, landmarks[263].y])
    mouth_left = np.array([landmarks[61].x, landmarks[61].y])
    mouth_right = np.array([landmarks[291].x, landmarks[291].y])
    nose_tip = np.array([landmarks[4].x, landmarks[4].y])

    # Calculate distances
    eye_distance = np.linalg.norm(left_eye - right_eye)
    mouth_width = np.linalg.norm(mouth_left - mouth_right)
    eye_mouth_distance = np.linalg.norm((left_eye + right_eye) / 2 - nose_tip)

    # Normalize features by face width
    eye_openness = eye_distance / face_width
    mouth_openness = mouth_width / face_width
    smile_width = mouth_width / face_width

    return eye_openness, mouth_openness, smile_width

# Function to calculate final metrics
def calculate_final_metrics(emotion_history):
    # Calculate average emotion probabilities
    avg_emotion_probabilities = np.mean(emotion_history, axis=0)

    # Calculate polarity score
    polarity_score = 0
    for i, prob in enumerate(avg_emotion_probabilities):
        emotion = emotion_labels[i]
        polarity_score += prob * emotion_polarity[emotion]

    # Calculate arousal level
    arousal_scores = {"High": 1, "Medium": 0, "Low": -1}
    for i, prob in enumerate(avg_emotion_probabilities):
        emotion = emotion_labels[i]
        arousal_scores[emotion_arousal[emotion]] += prob
    arousal_level = max(arousal_scores, key=arousal_scores.get)

    # Calculate mood stability index
    mood_stability_index = np.var(emotion_history, axis=0).mean()

    # Calculate dominance score (example: based on high-arousal emotions)
    dominance_score = np.mean([prob for i, prob in enumerate(avg_emotion_probabilities) if emotion_arousal[emotion_labels[i]] == "High"])

    return avg_emotion_probabilities, polarity_score, arousal_level, dominance_score, mood_stability_index

# Main loop for capturing frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Check if 20 seconds have passed since the last processing
    current_time = time.time()
    if current_time - last_processed_time >= 20:
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get face width (distance between eyes)
                left_eye = np.array([face_landmarks.landmark[33].x, face_landmarks.landmark[33].y])
                right_eye = np.array([face_landmarks.landmark[263].x, face_landmarks.landmark[263].y])
                face_width = np.linalg.norm(left_eye - right_eye)

                # Normalize facial features
                eye_openness, mouth_openness, smile_width = normalize_features(face_landmarks.landmark, face_width)

                # Resize frame for ViT model input
                resized_frame = cv2.resize(frame, (224, 224))
                inputs = feature_extractor(images=resized_frame, return_tensors="pt")

                # Predict emotion probabilities
                with torch.no_grad():
                    outputs = model(**inputs)
                    emotion_probabilities = torch.softmax(outputs.logits, dim=-1).numpy()[0]

                # Add to emotion history
                emotion_history.append(emotion_probabilities)

                # Get the dominant emotion
                dominant_emotion = emotion_labels[np.argmax(emotion_probabilities)]

                # Display dominant emotion and probabilities
                print("\n--- Emotion Probabilities for Last 20 Seconds ---")
                print(f"Dominant Emotion: {dominant_emotion}")
                for i, prob in enumerate(emotion_probabilities):
                    print(f"{emotion_labels[i]}: {prob:.2f}")
                print("-----------------------------------------------")

        # Update last processed time
        last_processed_time = current_time

    # Show the frame
    cv2.imshow("Real-Time Emotion Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate final metrics using the entire emotion history
if emotion_history:
    avg_emotion_probabilities, polarity_score, arousal_level, dominance_score, mood_stability_index = calculate_final_metrics(emotion_history)

    print("\n--- Final Metrics ---")
    print(f"Average Emotion Probabilities: {avg_emotion_probabilities}")
    print(f"Polarity Score: {polarity_score:.2f}")
    print(f"Arousal Level: {arousal_level}")
    print(f"Dominance Score: {dominance_score:.2f}")
    print(f"Mood Stability Index: {mood_stability_index:.2f}")
    print("---------------------")

# Cleanup
cap.release()
cv2.destroyAllWindows()