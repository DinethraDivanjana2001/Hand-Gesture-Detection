import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints, prob_viz

# Load model
model = load_model('action.h5')

# Actions and paths
actions = np.array(['hello', 'thanks', 'i love you'])
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]

# New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.7

# Access webcam
cap = cv2.VideoCapture(0)
with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])

            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np
