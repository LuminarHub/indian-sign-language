from django.shortcuts import render,redirect,get_object_or_404
from django.http import HttpResponse,JsonResponse
import cv2

import numpy as np
import math
import matplotlib.pyplot as plt
from django.contrib import messages
import time
from PIL import Image
from Sign_Application.models import Registration
from django.views.generic import TemplateView
from PIL import Image
import os
from django.core.mail import send_mail
import random
import string
from django.utils import timezone
from django.shortcuts import render
from django.http import HttpResponse,StreamingHttpResponse

from PIL import Image
import matplotlib.pyplot as plt
from django.contrib import messages
import matplotlib
from collections import Counter
from .models import *
matplotlib.use('Agg')

def Home(request):
    hom = request.session['Email']
    audio = Registration.objects.get(Email=hom)
    print(audio)
    au = audio.audio.url
    print("======",au)
    return render(request,"home.html",{'home':hom,'audio':au})

from django.shortcuts import render
from django.http import StreamingHttpResponse, HttpResponse
import cv2
import mediapipe as mp
import numpy as np
import time
from collections import Counter
from pathlib import Path
import torch
from TTS.api import TTS
from tensorflow.keras.models import load_model
import os
import os
import time
import simpleaudio as sa  # Install with `pip install simpleaudio`
from pydub import AudioSegment
from pydub.playback import play 
from django.core.files.base import ContentFile


def slow_down_audio(audio_path, factor=0.9):
        """Load audio, slow it down, and save it back"""
        audio = AudioSegment.from_wav(audio_path)  # Load audio
        new_frame_rate = 44100
        slowed_audio = audio.set_frame_rate(new_frame_rate)  # Adjust speed
        return slowed_audio
        # slowed_audio.export(audio_path, format="wav")

def reduce_noise(audio):
    """Apply a basic noise reduction by normalizing quiet parts"""
    return audio.remove_dc_offset().normalize()


class SignLanguageWithVoiceClone:
    def __init__(self, model_path, labels, reference_voice_path,user_id,
                 tts_model="tts_models/multilingual/multi-dataset/your_tts",
                 image_size=480, gesture_hold_time=1.0, 
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
        self.user_id = user_id
        # Initializing TTS
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Loading TTS model...")
        self.tts = TTS(model_name=tts_model, progress_bar=True, gpu=torch.cuda.is_available())
        
        # Initializing MediaPipe for hand landmarks detection
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
        
        # Flag to track if new audio is available
        self.new_audio_available = False
        self.current_audio_path = None

        # Create output directory
        self.output_dir = Path("static/audio")
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Initialize video capture
        self.video = cv2.VideoCapture(0)
        
    def process_hand_landmarks(self, img, results):
        """Process hand landmarks for gesture recognition."""
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
        """Predict sign language gesture from processed image."""
        if processed_img is None:
            return None, 0
            
        prediction = self.model.predict(processed_img, verbose=0)
        predicted_idx = np.argmax(prediction)
        confidence = np.max(prediction)
        
        if confidence > self.confidence_threshold:
            return self.labels[predicted_idx], confidence
        return None, confidence

    def process_prediction(self, prediction, confidence):
        """Process the prediction buffer and confirm gestures."""
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

    

    
    # Install with `pip install pydub ffmpeg`
    
    def clone_voice(self, text, output_filename=None):
        """Generate speech with cloned voice and play it correctly"""
        if not text:
            return None
                
        if output_filename is None:
            output_filename = f"cloned_{int(time.time())}.wav"
                
        output_path = self.output_dir / output_filename
        try:
            print(f"Cloning voice for: {text}")
            self.tts.tts_to_file(
                text=text,
                file_path=str(output_path),
                speaker_wav=self.reference_voice_path,
                language="en",
                peaker_embedding=True
            )
            
            print(f"Voice cloning completed! Output saved to {output_path}")
            # slow_down_audio(str(output_path), factor=0.9)
            # Ensure proper sample rate & play audio
            self.play_audio(str(output_path))

            self.current_audio_path = str(output_path)
            self.new_audio_available = True
            return str(output_path)
        except Exception as e:
            print(f"Error in voice cloning: {str(e)}")
            return None

    

    def play_audio(self, file_path):
        """Play the generated audio with the correct sample rate and enhancements"""
        if not os.path.exists(file_path):
            print(f"Audio file not found: {file_path}")
            return
        
        try:
            print(f"Playing audio: {file_path}")
            audio = AudioSegment.from_wav(file_path)

            # Set correct sample rate
            target_sample_rate = 44100
            if audio.frame_rate != target_sample_rate:
                print(f"Resampling audio from {audio.frame_rate}Hz to {target_sample_rate}Hz")
                audio = audio.set_frame_rate(target_sample_rate)

            # Ensure 16-bit PCM format (for better clarity)
            if audio.sample_width != 2:
                print("Converting audio to 16-bit PCM")
                audio = audio.set_sample_width(2)

            # Convert to mono if needed
            if audio.channels != 1:
                print("Converting audio to mono")
                audio = audio.set_channels(1)

            # Increase volume
            audio = audio + 5  

            # Save the processed audio to a file
            processed_audio_path = file_path.replace(".wav", "_processed.wav")
            audio.export(processed_audio_path, format="wav")

            # Save in the database
            try:
                reg = get_object_or_404(Registration, id=self.user_id)
                with open(processed_audio_path, "rb") as f:
                    audio_file = ContentFile(f.read(), name=processed_audio_path.split("/")[-1])
                    Audio.objects.create(audio_new=audio_file, user=reg)
                print(f"Processed audio saved to {processed_audio_path}")
            except Exception as e:
                print("Error saving audio to DB:", e)

            play(audio)

        except Exception as e:
            print(f"Error playing audio: {str(e)}")

    def get_frame(self):
        """Capture video frame and process sign language."""
        success, img = self.video.read()
        if not success:
            return None

        # Process hand landmarks and predict sign language
        results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        imgWhite, processed_img = self.process_hand_landmarks(img, results)

        confirmed_prediction = None
        confirmed_confidence = 0
        
        if processed_img is not None:
            predicted_char, confidence = self.predict_sign(processed_img)
            confirmed_prediction, confirmed_confidence = self.process_prediction(predicted_char, confidence)
            self.draw_interface(img, confirmed_prediction, confirmed_confidence)
        cv2.imshow("Sign Language with Voice Cloning", img)
        # Handle key press events
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # Space to finalize word
            if self.word_buffer:
                self.sentence_buffer.append(self.word_buffer)
                # Generate voice for the word
                self.clone_voice(
                    self.word_buffer,
                    f"word_{self.word_buffer}_{int(time.time())}.wav"
                )
                self.word_buffer = ""
                self.last_confirmed_prediction = None
        elif key == ord('.'):  # Period to finalize sentence
            if self.word_buffer:
                self.sentence_buffer.append(self.word_buffer)
            self.final_sentence = ' '.join(self.sentence_buffer) + '.'
            print(f"Final Sentence: {self.final_sentence}")
            
            # Generate speech with cloned voice for the full sentence
            if self.final_sentence:
                self.clone_voice(
                    self.final_sentence,
                    f"sentence_{int(time.time())}.wav"
                )

            self.sentence_buffer = []
            self.word_buffer = ""
            self.last_confirmed_prediction = None
        elif key == ord('b'):  # Backspace to remove last letter
            if self.word_buffer:
                self.word_buffer = self.word_buffer[:-1]
                self.last_confirmed_prediction = None
        elif key == ord('q'):  # Quit the application
            cv2.destroyAllWindows()
            self.video.release()
            return None

        # Encode the frame to send it via the HTTP response
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()

    def draw_interface(self, img, prediction=None, confidence=None):
        """Draw detected letter, confidence, word, and sentence on the video frame."""
        h, w, _ = img.shape

        # Create a semi-transparent overlay for better readability
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 120), (0, 0, 0), -1)  # Black transparent box
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)  # Transparency effect

        # Display detected letter
        if prediction and confidence:
            cv2.putText(img, f"Letter: {prediction} ({confidence:.2f})", 
                        (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 255, 0),  # Green color
                        2, 
                        cv2.LINE_AA)
        detected_word = ''.join(self.word_buffer)
        cv2.putText(img, f"Word: {detected_word}", 
                    (20, h - 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (255, 0, 0),  # Blue color
                    2, 
                    cv2.LINE_AA)

        # Show the full sentence at the very bottom
        full_sentence = ' '.join(self.sentence_buffer)
        cv2.putText(img, f"Sentence: {full_sentence}", 
                    (20, h - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 255),  # Yellow color
                    2, 
                    cv2.LINE_AA)

    def get_word(self):
        if self.word_buffer:
            latest_word = self.word_buffer[-1]
            print(f"Returning latest word: {latest_word}")
            return latest_word
        print("No word detected!")
        return None


    def get_current_audio(self):
        if self.new_audio_available and self.current_audio_path:
            audio_filename = os.path.basename(self.current_audio_path)
            print(f"Returning latest audio: {audio_filename}")
            return audio_filename
        print("No audio found!")
        return None 
    # def get_current_audio(self):
    #     if self.audio_buffer:
    #         latest_audio = self.audio_buffer[-1]
    #         print(f"Returning latest audio: {latest_audio}")
    #         return latest_audio
    #     print("No audio found!")
    #     return None


from .middleware import SessionMiddleware
# Create a global instance of the SignLanguageWithVoiceClone class
camera_instance = None
# email=
def get_camera_instance():
    global camera_instance
    
    # Check if the 'Email' is in the session
    email = SessionMiddleware.get_email()

    if email is None:
        raise ValueError("Email not found in session")

    if camera_instance is None:
        data = Registration.objects.get(Email=email)
        camera_instance = SignLanguageWithVoiceClone(
            model_path="D:/Projects/Indian Sign Language/Sign_Language/Sign_Application/best_sign_language_model.keras",  # Update path as needed
            labels=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"],
            reference_voice_path=data.audio.path,  # Update path as needed
            user_id = data.id
        )
    return camera_instance


from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
@require_http_methods(["POST"])
def handle_command(request):
    """Handle commands from the frontend to control the sign language detector."""
    action = request.GET.get('action', '')
    print("action",action)
    try:
        camera = get_camera_instance()
        if not camera:
            return JsonResponse({'success': False, 'error': 'Camera instance not available'}, status=500)
        print("==========",camera.word_buffer)
        # Simulate key presses based on action
        if action == 'space':
            print("working")
            # Finalize word (space key)
            if camera.word_buffer:
                camera.sentence_buffer.append(camera.word_buffer)
                # self.sentence_buffer.append(self.word_buffer)
                # Generate voice for the word
                camera.clone_voice(
                    camera.word_buffer,
                    f"word_{camera.word_buffer}_{int(time.time())}.wav"
                )
                camera.word_buffer = ""
                camera.last_confirmed_prediction = None
                
        elif action == 'period':
            # Finalize sentence (period key)
            if camera.word_buffer:
                camera.sentence_buffer.append(camera.word_buffer)
            camera.final_sentence = ' '.join(camera.sentence_buffer) + '.'
            
            # Generate speech with cloned voice for the full sentence
            if camera.final_sentence:
                camera.clone_voice(
                    camera.final_sentence,
                    f"sentence_{int(time.time())}.wav"
                )
            
            camera.sentence_buffer = []
            camera.word_buffer = ""
            camera.last_confirmed_prediction = None
            
        elif action == 'backspace':
            # Remove last letter (backspace key)
            if camera.word_buffer:
                camera.word_buffer = camera.word_buffer[:-1]
                camera.last_confirmed_prediction = None
                
        elif action == 'submit':
            # Custom action to submit the detection
            # You can add code here to save the detection to a database or perform other actions
            final_text = ' '.join(camera.sentence_buffer)
            if camera.word_buffer:
                final_text += ' ' + camera.word_buffer
                
            if final_text:
                # Here you would typically save the detection to a database
                # For now, we'll just return success
                return JsonResponse({'success': True, 'message': 'Detection submitted successfully', 'text': final_text})
            else:
                return JsonResponse({'success': False, 'error': 'No text to submit'})
        
        return JsonResponse({'success': True})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


import os
import logging

logger = logging.getLogger(__name__)

def hand_signal_detection(request):
    """Handles sign detection and speech synthesis."""
    camera = get_camera_instance()

    if not camera:
        logger.error("Camera instance is None.")
        return render(request, 'video.html', {'error': 'Camera not available'})

    # Fetch the detected word and audio file
    word = camera.get_word() if hasattr(camera, 'get_word') else None
    audio_file = camera.get_current_audio() if hasattr(camera, 'get_current_audio') else None

    # Debugging outputs
    logger.info(f"Detected word: {word}")
    logger.info(f"Audio file: {audio_file}")
    
    if not word:
        word = "No word detected"
    
    audio_path = os.path.join("static", "audio", audio_file) if audio_file else ""

    if audio_file and not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        audio_file = ""

    context = {
        'word': word,
        'sentence': ' '.join(getattr(camera, 'sentence_buffer', [])),
        'audio_file': audio_file
    }

    return render(request, 'video.html', context)

import cv2

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        
        
from django.views.decorators import gzip
@gzip.gzip_page
def video_feed(request):
    """Streaming HTTP response for live video feed."""
    try:
        # camera = get_camera_instance()
        # if not camera:
        #     return HttpResponse("Camera not available", status=500)
        email = SessionMiddleware.get_email()

        if email is None:
            raise ValueError("Email not found in session")

    
        data = Registration.objects.get(Email=email)
        return StreamingHttpResponse(gen(SignLanguageWithVoiceClone(
            model_path="D:/Projects/Indian Sign Language/Sign_Language/Sign_Application/best_sign_language_model.keras",  # Update path as needed
            labels=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"],
            reference_voice_path=data.audio.path,  # Update path as needed,
            user_id = data.id
        )), 
                                     content_type='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        return HttpResponse(f"Streaming error: {str(e)}", status=500)

def get_latest_data(request):
    """AJAX endpoint to get the latest word and audio file."""
    camera = get_camera_instance()
    if not camera:
        return JsonResponse({'error': 'Camera instance not available'}, status=500)

    word = camera.get_word() if hasattr(camera, 'get_word') else "No word detected"
    audio_file = camera.get_current_audio() if hasattr(camera, 'get_current_audio') else ""

    data = {
        'word': word,
        'sentence': ' '.join(getattr(camera, 'sentence_buffer', [])),
        'audio_file': audio_file
    }

    return JsonResponse(data)

def RegistrationForm(req):
    return render(req,"Registration.html")


def Registration_save(request):
    if request.method == "POST":
        try:
            nm = request.POST.get('uname')
            em = request.POST.get('email')
            passw = request.POST.get('password')
            con = request.POST.get('cpassword')
            
            # Check if password and confirm password match
            if passw != con:
                messages.error(request, "Password and confirm password do not match.")
                return redirect(RegistrationForm)
            
            # Creating and saving registration object
            registration = Registration(username=nm, Email=em, Password=passw, Confirm_Password=con)
            registration.save()
            
            messages.success(request, "Registered Successfully")
            return redirect(Login_Pg)

        except Exception as e:
            # Log the error (optional)
            print(f"Error occurred: {e}")
            messages.error(request, "An error occurred during registration. Please try again later.")
            return redirect(RegistrationForm)

def Login_Pg(req):
    return render(req,"Login_Pg.html")



def Login_fun(request):
    if request.method=="POST":
        nm=request.POST.get('email')
        pwd=request.POST.get('password')
        if Registration.objects.filter(Email=nm,Password=pwd).exists():
            request.session['Email']=nm
            request.session['Password']=pwd
            messages.success(request, "Logged in Successfully")
            return redirect(Home)
        else:
            messages.warning(request, "Check Your Credentials")
            return redirect(Login_Pg)
    else:
        messages.warning(request, "Check Your Credentials Or Sign Up ")
        return redirect(Login_Pg)

def Logout_fn(request):
    del request.session['Email']
    del request.session['Password']
    messages.success(request, "Logged Out Successfully")
    return redirect(Login_Pg)
            
            

import io
import base64


def show_imagess(request):
    if request.method == 'POST':
        user_input = request.POST.get('text', '').lower()
        valid_formats = ['png', 'jpg', 'jpeg']
        images = []
        fig, axs = plt.subplots(1, len(user_input), figsize=(len(user_input) * 4, 4))
        if len(user_input) == 1:
            axs = [axs]
        
        for i, char in enumerate(user_input):
            if char.isalnum():
                for img_format in valid_formats:
                    img_path = f"ASL/{char.lower()}.{img_format}"
                    
                    try:
                        img = Image.open(img_path)
                        img = img.convert('RGB')
                        axs[i].imshow(img)
                        axs[i].axis('off')  
                        buffer = io.BytesIO()
                        img.save(buffer, format='PNG')
                        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        images.append(image_base64)
                        
                        break  
                    except FileNotFoundError:
                        pass  
                    except Exception as e:
                        print(f"Error processing image for character {char}: {e}")
            elif char.isspace():
                space_img_path = f"ASL/space.jpg"
                try:
                    img = Image.open(space_img_path)
                    img = img.convert('RGB')                   
                    axs[i].imshow(img)
                    axs[i].axis('off')
                    
                    buffer = io.BytesIO()
                    img.save(buffer, format='PNG')
                    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    images.append(image_base64)
                except Exception as e:
                    print(f"Error processing space image: {e}")
        
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)  
        return render(request, 'text_to_sign.html', {'images': images, 'input_text': user_input})
    return render(request, 'text_to_sign.html')





def show_images(user_input):
    # Define valid image formats
    valid_formats = ['png', 'jpg', 'jpeg']

    # Create a subplot for each image
    fig, axs = plt.subplots(1, len(user_input), figsize=(len(user_input) * 4, 4))

    for i, char in enumerate(user_input):
        if char.isalnum():
            # Iterate through valid image formats
        
            for img_format in valid_formats:
                img_path = f"D:/Internship Luminar/Main Projects/Sign Language (American)/Sign_Language/ASL/{char.lower()}.{img_format}"

                try:
                    img = Image.open(img_path)
                    
                    axs[i].imshow(img)
                    axs[i].axis('off')  # Turn off axis for cleaner display
                    break  # Break loop if image is found
                except FileNotFoundError:
                    pass  # Continue to the next format if file not found
        elif char.isspace():
            # Display space image
            space_img_path = f"D:/Internship Luminar/Main Projects/Sign Language (American)/Sign_Language/ASL/space.jpg"
            img = Image.open(space_img_path)
            axs[i].imshow(img)
            axs[i].axis('off') 
    plt.show()




class Learning_page(TemplateView):
    template_name = "learning_page.html"

    

class Live(TemplateView):
    template_name = "video.html"
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user = self.request.session['Email']
        id = Registration.objects.get(Email = user)
        context['audio'] =Audio.objects.filter(user=id).last()
        print(Audio.objects.filter(user=id).first())
        return context


class Text_to_SignLanguage(TemplateView):
    template_name='text_to_sign.html'


def forgetpassword_enteremail(request):
    if (request.method == "POST"):
        email = request.POST['email']

        try:
            user=Registration.objects.get(Email=email)
        except Registration.DoesNotExist:
            user=[]

        if user:
            # Generate a random 6-character OTP
            otp = ''.join(random.choices(string.digits, k=6))

            # Store OTP and timestamp in the session
            request.session['otp'] = otp
            request.session['otp_timestamp'] = timezone.now().timestamp()
            request.session['email'] = email

            # Send the OTP to the user's email
            subject = 'Password Reset OTP'
            message = f'Your OTP for password reset is: {otp}.valid upto 5 minutes'
            from_email = 'jipsongeorge753@gmail.com'
            to_email = email
            send_mail(subject, message, from_email, [to_email], fail_silently=False)
            return redirect('otp')
        else:
            messages.error(request, "invalid user email")

    return render(request,'forget_password_enter_email.html')

def otp(request):
    if (request.method == "POST"):
        otp= request.POST['otp']
        stored_otp = request.session.get('otp')
        stored_email = request.session.get('email')
        timestamp = request.session.get('otp_timestamp')
        print(timestamp)

        # Verify the OTP
        if stored_otp == otp and timezone.now().timestamp() - timestamp <= 300:
            # OTP is valid and not expired (within 5 minutes)
            # You can add additional logic here if needed
            return redirect('new_password')
        elif timestamp is not None:
            if timezone.now().timestamp() - timestamp >= 300:
                request.session.pop('otp', None)
                request.session.pop('otp_timestamp', None)
                request.session.pop('email', None)
                messages.error(request, "invalid OTP")

        else:
            # Clear the session data if OTP is invalid or expired

            messages.error(request, "invalid OTP")

    return render(request,'forget_password_enter_otp.html')


def newpassword(request):
    if (request.method == "POST"):
        new_password= request.POST['new_password']
        # email = request.data.get('email')
        # otp = request.data.get('otp')


        # if not email or not new_password or not otp:
        #     return Response({'error': 'email and new_password or otp are required'}, status=status.HTTP_400_BAD_REQUEST)

        stored_otp = request.session.get('otp')
        stored_email = request.session.get('email')
        timestamp = request.session.get('otp_timestamp')

        # if stored_otp == otp and stored_email == email:

        user = Registration.objects.get(Email=stored_email)
        if user is None:
            messages.error(request, "invalid user")

        # Change the user's password
        user.Password=new_password
        user.Confirm_Password=new_password
        user.save()

        # Clear the session data after successful password change
        request.session.pop('otp', None)
        request.session.pop('otp_timestamp', None)
        request.session.pop('email', None)

        return redirect('Login_Pg')

    return render(request, 'new_password.html')



def add_audio(request):
    """View for adding a new address"""
    
    if request.method == 'POST':

        if 'audio' not in request.FILES:
            return JsonResponse({'success': False, 'message': 'No audio file provided'}, status=400)

        audio_file = request.FILES['audio']
        
        try:
            data = Registration.objects.get(Email=request.session['Email'])
            data.audio=audio_file
            data.save()
            messages.success(request, 'Address added successfully!')
            return redirect('Home')
        except Exception as e:
            print("Error:", str(e))
            return JsonResponse({'success': False, 'message': f'Error adding address: {str(e)}'}, status=500)

    return JsonResponse({'success': False, 'message': 'Invalid request method'}, status=405)
    
    
class MainHomeView(TemplateView):
    template_name = "mainhome.html"