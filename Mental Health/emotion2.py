"""input webcam with no descriptive measures only objective metrics found in the code"""


import cv2
import mediapipe as mp
import numpy as np
import time
from transformers import ViTForImageClassification, ViTFeatureExtractor
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Load a pre-trained Vision Transformer (ViT) model for emotion recognition
model_name = "google/vit-base-patch16-224"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

# Load the model with 22 output classes and ignore size mismatch
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=22,  # Set the number of output classes to 22
    ignore_mismatched_sizes=True  # Ignore size mismatch in the output layer
)

# Emotion labels for 22 classes
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
    "Angry": "High", "Disgust": "Medium", "Fear": "High", "Happy": "Medium", "Sad": "Low",
    "Surprise": "High", "Neutral": "Low", "Contempt": "Medium", "Anxiety": "High",
    "Calm": "Low", "Confusion": "Medium", "Desire": "Medium", "Empathy": "Medium",
    "Excitement": "High", "Interest": "Medium", "Pride": "Medium", "Relief": "Medium",
    "Shame": "Low", "Satisfaction": "Medium", "Boredom": "Low", "Amusement": "Medium", "Awe": "Medium"
}

# Initialize MediaPipe FaceMesh for facial landmark detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide video file path

# Variables for storing emotion data
emotion_history = []
start_time = time.time()
last_processed_time = start_time
current_emotion_probs = None
dominant_emotion = "Processing..."
current_frame_count = 0
total_frames = 0

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

# Function to create a bar chart of emotions
def create_emotion_chart(emotion_probs):
    # Get top 5 emotions for display
    top_indices = np.argsort(emotion_probs)[-5:][::-1]
    top_emotions = [emotion_labels[i] for i in top_indices]
    top_probs = [emotion_probs[i] * 100 for i in top_indices]  # Convert to percentage
    
    # Create figure
    fig = Figure(figsize=(3,1.5), dpi=100)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    
    # Create horizontal bar chart
    bars = ax.barh(top_emotions, top_probs, color='skyblue')
    
    # Add percentage labels
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                ha='left', va='center')
    
    ax.set_xlim(0, 100)
    ax.set_title('Top 5 Detected Emotions')
    fig.tight_layout()
    
    # Convert plot to image
    canvas.draw()
    chart_image = np.array(canvas.renderer.buffer_rgba())
    chart_image = cv2.cvtColor(chart_image, cv2.COLOR_RGBA2BGR)
    
    return chart_image

# Function to calculate final metrics
def calculate_final_metrics(emotion_history):
    # Calculate average emotion probabilities
    avg_emotion_probabilities = np.mean(emotion_history, axis=0)
    
    # 1. Emotion Intensity Score (0-100%)
    emotion_intensity = {emotion_labels[i]: prob * 100 for i, prob in enumerate(avg_emotion_probabilities)}
    
    # 2. Emotion Polarity Score (-1 to +1)
    polarity_score = 0
    for i, prob in enumerate(avg_emotion_probabilities):
        emotion = emotion_labels[i]
        polarity_score += prob * emotion_polarity[emotion]
    
    # 3. Arousal Level (Low, Medium, High)
    arousal_scores = {"High": 0, "Medium": 0, "Low": 0}
    for i, prob in enumerate(avg_emotion_probabilities):
        emotion = emotion_labels[i]
        arousal_scores[emotion_arousal[emotion]] += prob
    arousal_level = max(arousal_scores, key=arousal_scores.get)
    
    # 4. Dominance Score (-1 to +1)
    # Calculate based on specific emotions associated with dominance
    dominant_emotions = ["Angry", "Pride", "Excitement", "Satisfaction"]
    submissive_emotions = ["Fear", "Shame", "Anxiety", "Confusion"]
    
    dominance_score = 0
    for emotion, score in zip(emotion_labels, avg_emotion_probabilities):
        if emotion in dominant_emotions:
            dominance_score += score
        elif emotion in submissive_emotions:
            dominance_score -= score
    dominance_score = max(min(dominance_score, 1), -1)  # Clamp between -1 and 1
    
    # 5. Mood Stability Index (Stable, Unstable, Highly Unstable)
    mood_variance = np.var(emotion_history, axis=0).mean()
    if mood_variance < 0.02:
        mood_stability = "Stable"
    elif mood_variance < 0.05:
        mood_stability = "Unstable"
    else:
        mood_stability = "Highly Unstable"
    
    # 6. Mental Health Indicator (Normal, At Risk, Severe)
    # Calculate based on combination of negative emotions and variance
    negative_emotions = ["Angry", "Disgust", "Fear", "Sad", "Anxiety", "Shame", "Confusion"]
    negative_score = sum(avg_emotion_probabilities[i] for i, emotion in enumerate(emotion_labels) 
                         if emotion in negative_emotions)
    
    if negative_score < 0.3 and mood_variance < 0.03:
        mental_health = "Normal"
    elif negative_score < 0.5 and mood_variance < 0.05:
        mental_health = "At Risk"
    else:
        mental_health = "Severe"
    
    return {
        "emotion_intensity": emotion_intensity,
        "polarity_score": polarity_score,
        "arousal_level": arousal_level,
        "dominance_score": dominance_score,
        "mood_stability": mood_stability,
        "mental_health": mental_health
    }

# Create window
cv2.namedWindow("Mental Health Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Mental Health Detection", 1280, 720)

# Main loop for capturing frames
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video reached")
            break
        
        current_frame_count += 1
        total_frames += 1
        
        # Create a copy of the frame for display
        display_frame = frame.copy()
        
        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        # Draw face mesh landmarks if detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing = mp.solutions.drawing_utils
                mp_drawing_styles = mp.solutions.drawing_styles
                mp_drawing.draw_landmarks(
                    display_frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
        
        # Check if 20 seconds have passed since the last processing
        current_time = time.time()
        if current_time - last_processed_time >= 20 or current_emotion_probs is None:
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Resize frame for ViT model input
                    resized_frame = cv2.resize(frame, (224, 224))
                    inputs = feature_extractor(images=resized_frame, return_tensors="pt")
                    
                    # Predict emotion probabilities
                    with torch.no_grad():
                        outputs = model(**inputs)
                        emotion_probabilities = torch.softmax(outputs.logits, dim=-1).numpy()[0]
                    
                    # Add to emotion history
                    emotion_history.append(emotion_probabilities)
                    
                    # Update current emotion probabilities
                    current_emotion_probs = emotion_probabilities
                    
                    # Get the dominant emotion
                    dominant_emotion = emotion_labels[np.argmax(emotion_probabilities)]
                    
                    print(f"\n--- Emotion Analysis at {current_time - start_time:.1f}s ---")
                    print(f"Dominant Emotion: {dominant_emotion}")
            
            # Update last processed time
            last_processed_time = current_time
        
        # Display the emotion chart on the frame if we have emotions detected
        if current_emotion_probs is not None:
            # Create emotion chart
            chart_img = create_emotion_chart(current_emotion_probs)
            
            # Get dimensions
            chart_h, chart_w = chart_img.shape[:2]
            frame_h, frame_w = display_frame.shape[:2]
            
            # Calculate position (bottom right corner)
            x_offset = frame_w - chart_w - 10
            y_offset = frame_h - chart_h - 10
            
            # Ensure the chart fits within frame bounds
            if x_offset >= 0 and y_offset >= 0:
                # Create ROI and overlay chart
                roi = display_frame[y_offset:y_offset+chart_h, x_offset:x_offset+chart_w]
                
                # Create a semi-transparent background
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (x_offset, y_offset), 
                             (x_offset+chart_w, y_offset+chart_h), (0, 0, 0), -1)
                display_frame = cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0)
                
                # Now place the chart
                display_frame[y_offset:y_offset+chart_h, x_offset:x_offset+chart_w] = chart_img
            
            # Add dominant emotion text
            cv2.putText(display_frame, f"Dominant: {dominant_emotion}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display current analysis period
            time_elapsed = current_time - last_processed_time
            time_remaining = max(0, 20 - time_elapsed)
            cv2.putText(display_frame, f"Next update in: {time_remaining:.1f}s", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show the frame
        cv2.imshow("Mental Health Detection", display_frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Processing interrupted by user")
finally:
    # Calculate final metrics using the entire emotion history
    if emotion_history:
        final_metrics = calculate_final_metrics(emotion_history)
        
        print("\n=== FINAL MENTAL HEALTH METRICS ===")
        print(f"Total frames processed: {total_frames}")
        print(f"Total emotion samples: {len(emotion_history)}")
        print("\n1. Emotion Intensity Scores (0-100%):")
        # Sort emotions by intensity and show top 5
        sorted_emotions = sorted(final_metrics["emotion_intensity"].items(), 
                               key=lambda x: x[1], reverse=True)
        for emotion, intensity in sorted_emotions[:5]:
            print(f"   {emotion}: {intensity:.1f}%")
        
        print(f"\n2. Emotion Polarity Score (-1 to +1): {final_metrics['polarity_score']:.2f}")
        if final_metrics['polarity_score'] > 0.3:
            print("   Interpretation: Predominantly positive emotions")
        elif final_metrics['polarity_score'] < -0.3:
            print("   Interpretation: Predominantly negative emotions")
        else:
            print("   Interpretation: Mixed or neutral emotions")
        
        print(f"\n3. Arousal Level: {final_metrics['arousal_level']}")
        
        print(f"\n4. Dominance Score (-1 to +1): {final_metrics['dominance_score']:.2f}")
        if final_metrics['dominance_score'] > 0.3:
            print("   Interpretation: High confidence/control")
        elif final_metrics['dominance_score'] < -0.3:
            print("   Interpretation: Low confidence/submissive")
        else:
            print("   Interpretation: Balanced confidence level")
        
        print(f"\n5. Mood Stability Index: {final_metrics['mood_stability']}")
        
        print(f"\n6. Mental Health Indicator: {final_metrics['mental_health']}")
        print("==============================")
        
        # Create a final summary image
        summary_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Add title
        cv2.putText(summary_img, "MENTAL HEALTH ANALYSIS SUMMARY", 
                  (320, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Add metrics
        metrics_text = [
            f"Emotion Polarity: {final_metrics['polarity_score']:.2f}",
            f"Arousal Level: {final_metrics['arousal_level']}",
            f"Dominance Score: {final_metrics['dominance_score']:.2f}",
            f"Mood Stability: {final_metrics['mood_stability']}",
            f"Mental Health: {final_metrics['mental_health']}"
        ]
        
        for i, text in enumerate(metrics_text):
            cv2.putText(summary_img, text, (100, 150 + i*60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Show the top 5 emotions
        cv2.putText(summary_img, "Top 5 Emotions:", (700, 150), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        for i, (emotion, intensity) in enumerate(sorted_emotions[:5]):
            cv2.putText(summary_img, f"{emotion}: {intensity:.1f}%", 
                      (700, 200 + i*50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Display the summary
        cv2.imshow("Mental Health Analysis Summary", summary_img)
        cv2.waitKey(0)
        
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()