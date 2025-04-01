import cv2
import mediapipe as mp
import numpy as np
import torch
from tensorflow.keras.models import load_model
from TTS.api import TTS
import time
from collections import Counter
import os
from pathlib import Path

class SignLanguageWithVoiceClone:
    def __init__(self, model_path, labels, reference_voice_path,
                 tts_model="tts_models/multilingual/multi-dataset/your_tts",
                 image_size=224, gesture_hold_time=1.0, 
                 prediction_buffer_size=5, confidence_threshold=0.6):
        # Initializing sign language model and labels
        self.model = load_model(model_path)
        self.labels = labels
        self.image_size = image_size
        self.gesture_hold_time = gesture_hold_time
        self.prediction_buffer_size = prediction_buffer_size
        self.confidence_threshold = confidence_threshold
        
        # Storing reference voice path
        self.reference_voice_path = reference_voice_path
        
        # Initializing TTS 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Loading TTS model...")
        self.tts = TTS(model_name=tts_model, progress_bar=True, gpu=torch.cuda.is_available())
        
        # Initializing MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize buffers
        self.word_buffer = ""
        self.sentence_buffer = []
        self.final_sentence = ""
        self.prediction_buffer = []
        self.current_gesture_start = None
        self.last_confirmed_prediction = None
        self.stable_prediction_start = None
        self.prediction_history = []

        # Creating output directory 
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

    def process_hand_landmarks(self, img, results):
        h, w, _ = img.shape
        imgWhite = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8) * 255
        
        if not results.multi_hand_landmarks:
            return imgWhite, None
            
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
            
            # Draw landmarks on white canvas
            for point in landmarks:
                cv2.circle(
                    imgWhite,
                    (int(point[0] * self.image_size / w), int(point[1] * self.image_size / h)),
                    5, (0, 0, 255), cv2.FILLED
                )
            
            # Draw connections
            for connection in self.mp_hands.HAND_CONNECTIONS:
                start_idx, end_idx = connection
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]
                cv2.line(
                    imgWhite,
                    (int(start_point[0] * self.image_size / w), int(start_point[1] * self.image_size / h)),
                    (int(end_point[0] * self.image_size / w), int(end_point[1] * self.image_size / h)),
                    (0, 255, 0), 2
                )
                
        img_resized = cv2.resize(imgWhite, (224, 224))
        img_resized = img_resized / 255.0
        img_resized = np.expand_dims(img_resized, axis=0)
        
        return imgWhite, img_resized

    def predict_sign(self, processed_img):
        if processed_img is None:
            return None, 0
        
        prediction = self.model.predict(processed_img, verbose=0)
        predicted_idx = np.argmax(prediction)
        confidence = np.max(prediction)
        
        print(f"Prediction: {self.labels[predicted_idx]}, Confidence: {confidence}")  # Debugging print
        
        if confidence > self.confidence_threshold:
            return self.labels[predicted_idx], confidence
        return None, confidence


    def process_prediction(self, prediction, confidence):
        current_time = time.time()
        
        if prediction:
            self.prediction_buffer.append((prediction, confidence, current_time))
            self.prediction_buffer = [p for p in self.prediction_buffer 
                                    if current_time - p[2] <= self.gesture_hold_time]
            
            if len(self.prediction_buffer) >= 3:
                recent_predictions = [p[0] for p in self.prediction_buffer]
                counter = Counter(recent_predictions)
                most_common = counter.most_common(1)[0]
                
                if (most_common[1] >= len(self.prediction_buffer) * 0.7 and
                    most_common[0] != self.last_confirmed_prediction):
                    
                    if self.stable_prediction_start is None:
                        self.stable_prediction_start = current_time
                    elif current_time - self.stable_prediction_start >= self.gesture_hold_time:
                        self.last_confirmed_prediction = most_common[0]
                        self.word_buffer += most_common[0]
                        self.stable_prediction_start = None
                        self.prediction_buffer = []
                        return most_common[0], confidence
                
                elif self.stable_prediction_start is not None:
                    self.stable_prediction_start = None
        
        return None, 0

    def clone_voice(self, text, output_filename="output.wav"):
        """Generate speech with cloned voice"""
        output_path = self.output_dir / output_filename
        try:
            print(f"Cloning voice for: {text}")
            self.tts.tts_to_file(
                text=text,
                file_path=str(output_path),
                speaker_wav=self.reference_voice_path,
                language="en"
            )
            print(f"Voice cloning completed! Output saved to {output_path}")
            return str(output_path)
        except Exception as e:
            print(f"Error in voice cloning: {str(e)}")
            return None

    def draw_interface(self, imgWhite, prediction=None, confidence=None):
        if prediction and confidence:
            cv2.putText(imgWhite, f"{prediction} ({confidence:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        if self.stable_prediction_start:
            progress = min(1.0, (time.time() - self.stable_prediction_start) / self.gesture_hold_time)
            bar_width = int(self.image_size * 0.8)
            bar_height = 20
            bar_x = int(self.image_size * 0.1)
            bar_y = 50
            
            cv2.rectangle(imgWhite, (bar_x, bar_y), 
                        (bar_x + bar_width, bar_y + bar_height),
                        (200, 200, 200), -1)
            
            progress_width = int(bar_width * progress)
            cv2.rectangle(imgWhite, (bar_x, bar_y),
                        (bar_x + progress_width, bar_y + bar_height),
                        (0, 255, 0), -1)
        
        # Display the current word and sentence
        cv2.putText(imgWhite, f"Word: {self.word_buffer}",
                (10, self.image_size - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(imgWhite, f"Sentence: {' '.join(self.sentence_buffer)}",
                (10, self.image_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not open camera")

        print("/nControls:")
        print("Space - Finalize word")
        print(". - Finalize and speak sentence with cloned voice")
        print("b - Backspace")
        print("q - Quit")

        try:
            while True:
                success, img = cap.read()
                if not success:
                    print("Failed to capture frame")
                    break

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = self.hands.process(img_rgb)

                imgWhite, processed_img = self.process_hand_landmarks(img, results)

                if processed_img is not None:
                    predicted_char, confidence = self.predict_sign(processed_img)
                    confirmed_prediction, confirmed_confidence = self.process_prediction(predicted_char, confidence)
                    self.draw_interface(imgWhite, confirmed_prediction, confirmed_confidence)
                else:
                    self.draw_interface(imgWhite)

                # Concatenate the images side-by-side (horizontally)
                combined_frame = np.hstack((img, imgWhite))

                cv2.imshow("Sign Language with Voice Cloning", combined_frame)

                key = cv2.waitKey(20) & 0xFF

                if key == ord(' '):  # Space to finalize word
                    if self.word_buffer:
                        self.sentence_buffer.append(self.word_buffer)
                        self.word_buffer = ""
                        self.last_confirmed_prediction = None
                elif key == ord('.'):  # Period to finalize sentence
                    if self.word_buffer:
                        self.sentence_buffer.append(self.word_buffer)
                    self.final_sentence = ' '.join(self.sentence_buffer) + '.'
                    print(f"Final Sentence: {self.final_sentence}")

                    # Generate speech with cloned voice
                    if self.final_sentence:
                        output_file = self.clone_voice(
                            self.final_sentence,
                            f"cloned_sentence_{int(time.time())}.wav"
                        )

                    self.sentence_buffer = []
                    self.word_buffer = ""
                    self.last_confirmed_prediction = None
                elif key == ord('b'):  # Backspace
                    if self.word_buffer:
                        self.word_buffer = self.word_buffer[:-1]
                        self.last_confirmed_prediction = None
                elif key == ord('q'):  # Quit
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    
    reference_audio = input("Enter path to reference audio file (WAV format): ")
    
    # Initialize interpreter with  model and labels
    interpreter = SignLanguageWithVoiceClone(
        model_path = "D:/Projects/Indian Sign Language/FINAL_ISL/sign_language_model.keras",
        labels=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"],
        reference_voice_path=reference_audio,
        image_size=480,
        gesture_hold_time=1.0,
        prediction_buffer_size=5,
        confidence_threshold=0.6
    )
    
    try:
        interpreter.run()
    except Exception as e:
        print(f"Application error: {str(e)}")
        print("/nTroubleshooting tips:")
        print("1. Make sure you have installed all required packages")
        print("2. Ensure the reference audio file exists and is in WAV format")
        print("3. Check that you have enough disk space")
        print("4. Make sure you have a working webcam")
        print("5. Verify that your sign language model file exists")

if __name__ == "__main__":
    main()