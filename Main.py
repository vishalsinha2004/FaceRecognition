import cv2
import datetime
import numpy as np
import os
import time
from collections import deque

# Load cascades
face_cap = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

try:
    eye_cap = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
except:
    eye_cap = None
    print("Eye detection disabled (haarcascade_eye.xml not found)")

try:
    smile_cap = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
except:
    smile_cap = None
    print("Smile detection disabled (haarcascade_smile.xml not found)")

# Initialize video capture
video_cap = cv2.VideoCapture(0)
if not video_cap.isOpened():
    print("Error: Could not open video capture")
    exit()

# Create directory for saved files
os.makedirs("face_detection_output", exist_ok=True)

# Variables for FPS calculation
start_time = datetime.datetime.now()
frame_count = 0
fps = 0
fps_history = deque(maxlen=100)  # Store last 100 FPS values

# Variables for face counting
face_count = 0
prev_face_count = 0
total_faces_detected = 0
unique_faces = set()  # For simple face tracking

# Variables for recording
is_recording = False
video_writer = None
recording_start_time = None

# Variables for modes
gray_mode = False
blur_faces = False
age_gender_mode = False
mirror_mode = True
edge_detection = False
motion_detection = False
night_mode = False
face_tracking = False
emotion_mode = False
zoom_mode = False
zoom_factor = 1.0
max_zoom = 3.0

# Variables for motion detection
prev_frame = None
motion_threshold = 1000
motion_boxes = []

# Variables for face tracking
tracked_faces = {}
next_face_id = 1
track_colors = {}

# Variables for emotion simulation
emotions = ["Happy", "Neutral", "Surprised", "Angry"]
emotion_colors = {
    "Happy": (0, 255, 0),
    "Neutral": (255, 255, 255),
    "Surprised": (255, 255, 0),
    "Angry": (0, 0, 255)
}

# Variables for night mode
night_brightness = 50
night_contrast = 50

# Variables for face landmarks (simulated)
landmark_mode = False

# Variables for time tracking
program_start_time = time.time()

def adjust_brightness_contrast(img, brightness=0, contrast=0):
    """
    Adjust brightness and contrast of an image
    """
    brightness = max(-100, min(100, brightness))
    contrast = max(-100, min(100, contrast))
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        img = cv2.addWeighted(img, alpha_b, img, 0, gamma_b)
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        img = cv2.addWeighted(img, alpha_c, img, 0, gamma_c)
    
    return img

def get_track_color(face_id):
    """Get a consistent color for each tracked face"""
    if face_id not in track_colors:
        track_colors[face_id] = (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255)
        )
    return track_colors[face_id]

def detect_motion(current_frame, threshold):
    """Detect motion between frames"""
    global prev_frame, motion_boxes
    
    if prev_frame is None:
        prev_frame = current_frame
        return []
    
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    # Compute difference and threshold
    frame_diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
    
    # Dilate the threshold image
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    motion_boxes = []
    for contour in contours:
        if cv2.contourArea(contour) > threshold:
            (x, y, w, h) = cv2.boundingRect(contour)
            motion_boxes.append((x, y, w, h))
    
    prev_frame = current_frame.copy()
    return motion_boxes

def track_faces(faces):
    """Simple face tracking with ID assignment"""
    global tracked_faces, next_face_id
    
    current_face_ids = []
    updated_tracked_faces = {}
    
    for (x, y, w, h) in faces:
        # Find if this face matches any existing face
        matched_id = None
        for face_id, (fx, fy, fw, fh) in tracked_faces.items():
            # Simple distance-based matching
            distance = np.sqrt((x - fx)**2 + (y - fy)**2)
            if distance < 50:  # Threshold for matching
                matched_id = face_id
                break
        
        if matched_id is not None:
            updated_tracked_faces[matched_id] = (x, y, w, h)
            current_face_ids.append(matched_id)
        else:
            # New face detected
            updated_tracked_faces[next_face_id] = (x, y, w, h)
            current_face_ids.append(next_face_id)
            next_face_id += 1
    
    tracked_faces = updated_tracked_faces
    return tracked_faces

def draw_landmarks(image, face):
    """Simulate facial landmarks"""
    x, y, w, h = face
    center_x = x + w // 2
    center_y = y + h // 2
    
    # Eyes
    eye_y = y + h // 3
    left_eye = (center_x - w // 4, eye_y)
    right_eye = (center_x + w // 4, eye_y)
    cv2.circle(image, left_eye, 5, (0, 255, 255), -1)
    cv2.circle(image, right_eye, 5, (0, 255, 255), -1)
    
    # Mouth
    mouth_y = y + 2 * h // 3
    mouth_start = (center_x - w // 4, mouth_y)
    mouth_end = (center_x + w // 4, mouth_y)
    cv2.line(image, mouth_start, mouth_end, (0, 255, 255), 2)
    
    # Nose
    nose_tip = (center_x, center_y)
    cv2.circle(image, nose_tip, 3, (0, 255, 255), -1)

while True:
    ret, video_data = video_cap.read()
    if not ret:
        print("Error: Could not read frame")
        break
        
    # Apply mirror effect if enabled
    if mirror_mode:
        video_data = cv2.flip(video_data, 1)
    
    # Apply night mode adjustments
    if night_mode:
        video_data = adjust_brightness_contrast(video_data, night_brightness, night_contrast)
    
    # Store original frame for zooming
    original_frame = video_data.copy()
    
    # Apply zoom if enabled
    if zoom_mode and zoom_factor > 1.0:
        h, w = video_data.shape[:2]
        center_x, center_y = w // 2, h // 2
        zoom_w = int(w / zoom_factor)
        zoom_h = int(h / zoom_factor)
        video_data = video_data[
            max(0, center_y - zoom_h//2):min(h, center_y + zoom_h//2),
            max(0, center_x - zoom_w//2):min(w, center_x + zoom_w//2)
        ]
        video_data = cv2.resize(video_data, (w, h))
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cap.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    # Update face count statistics
    face_count = len(faces)
    total_faces_detected += max(0, face_count - prev_face_count)
    if face_count != prev_face_count:
        print(f"Number of faces detected: {face_count}")
        prev_face_count = face_count
    
    # Track faces if enabled
    if face_tracking:
        tracked_faces = track_faces(faces)
    
    # Detect motion if enabled
    if motion_detection:
        motion_boxes = detect_motion(video_data, motion_threshold)
    
    # Process each detected face
    for i, (x, y, w, h) in enumerate(faces):
        face_id = i + 1  # Default ID if not tracking
        
        # Get face ID if tracking is enabled
        if face_tracking:
            for fid, (fx, fy, fw, fh) in tracked_faces.items():
                if abs(x - fx) < 50 and abs(y - fy) < 50:  # Simple matching
                    face_id = fid
                    break
        
        # Blur face if enabled
        if blur_faces:
            face_roi = video_data[y:y+h, x:x+w]
            blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
            video_data[y:y+h, x:x+w] = blurred_face
        else:
            # Draw rectangle around the face
            color = (0, 255, 0)  # Default green
            if face_tracking:
                color = get_track_color(face_id)
            
            cv2.rectangle(video_data, (x, y), (x + w, y + h), color, 2)
            
            # Display face ID if tracking
            if face_tracking:
                cv2.putText(video_data, f"ID: {face_id}", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Detect eyes if available
        if eye_cap is not None:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cap.detectMultiScale(roi_gray, 1.1, 3)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(video_data, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 0, 0), 1)
        
        # Detect smiles if available
        if smile_cap is not None:
            roi_gray = gray[y:y+h, x:x+w]
            smiles = smile_cap.detectMultiScale(roi_gray, 1.8, 20)
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(video_data, (x+sx, y+sy), (x+sx+sw, y+sy+sh), (0, 0, 255), 1)
                cv2.putText(video_data, "Smiling", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Simulate age/gender prediction
        if age_gender_mode:
            cv2.putText(video_data, "Age: ~25", (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(video_data, "Gender: Male", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # Simulate emotion detection
        if emotion_mode:
            emotion = np.random.choice(emotions)
            cv2.putText(video_data, f"Emotion: {emotion}", (x, y-70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, emotion_colors[emotion], 1)
        
        # Draw simulated landmarks
        if landmark_mode:
            draw_landmarks(video_data, (x, y, w, h))
    
    # Highlight motion areas
    if motion_detection and motion_boxes:
        for (mx, my, mw, mh) in motion_boxes:
            cv2.rectangle(video_data, (mx, my), (mx+mw, my+mh), (255, 0, 255), 2)
    
    # Apply edge detection if enabled
    if edge_detection:
        edges = cv2.Canny(video_data, 100, 200)
        video_data = cv2.addWeighted(video_data, 0.7, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.3, 0)
    
    # Calculate and display FPS
    frame_count += 1
    if frame_count >= 10:
        end_time = datetime.datetime.now()
        delta = end_time - start_time
        fps = 10 / delta.total_seconds()
        fps_history.append(fps)
        start_time = end_time
        frame_count = 0
    
    # Display FPS information
    avg_fps = np.mean(fps_history) if fps_history else fps
    cv2.putText(video_data, f"FPS: {fps:.1f} (Avg: {avg_fps:.1f})", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display face count information
    cv2.putText(video_data, f"Faces: {face_count} (Total: {total_faces_detected})", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display current time and program duration
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    duration = time.time() - program_start_time
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    duration_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
    cv2.putText(video_data, current_time, (10, video_data.shape[0] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(video_data, f"Duration: {duration_str}", (10, video_data.shape[0] - 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display recording status
    if is_recording:
        recording_time = time.time() - recording_start_time if recording_start_time else 0
        rec_h, rec_m, rec_s = [int(x) for x in time.strftime("%H:%M:%S", time.gmtime(recording_time)).split(':')]
        rec_str = f"REC {rec_h:02d}:{rec_m:02d}:{rec_s:02d}"
        
        cv2.putText(video_data, rec_str, (video_data.shape[1] - 120, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.circle(video_data, (video_data.shape[1] - 30, 20), 10, (0, 0, 255), -1)
    
    # Display mode status
    mode_lines = [
        f"Blur: {'ON' if blur_faces else 'OFF'} | Gray: {'ON' if gray_mode else 'OFF'}",
        f"Age/Gender: {'ON' if age_gender_mode else 'OFF'} | Emotion: {'ON' if emotion_mode else 'OFF'}",
        f"Motion: {'ON' if motion_detection else 'OFF'} | Tracking: {'ON' if face_tracking else 'OFF'}",
        f"Edges: {'ON' if edge_detection else 'OFF'} | Night: {'ON' if night_mode else 'OFF'}",
        f"Zoom: {'ON' if zoom_mode else 'OFF'} (x{zoom_factor:.1f}) | Mirror: {'ON' if mirror_mode else 'OFF'}"
    ]
    
    for i, line in enumerate(mode_lines):
        cv2.putText(video_data, line, (10, 90 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Convert to grayscale if enabled
    if gray_mode:
        video_data = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
        video_data = cv2.cvtColor(video_data, cv2.COLOR_GRAY2BGR)
    
    # Write frame to video if recording
    if is_recording:
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join("face_detection_output", f"recording_{timestamp}.avi")
            video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (video_data.shape[1], video_data.shape[0]))
            recording_start_time = time.time()
        video_writer.write(video_data)
    
    # Show the frame
    cv2.imshow("Advanced Face Detection", video_data)
    
    # Key controls
    key = cv2.waitKey(10)
    if key == ord("a"):  # Exit
        break
    elif key == ord("s"):  # Save screenshot
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join("face_detection_output", f"face_detection_{timestamp}.jpg")
        cv2.imwrite(filename, video_data)
        print(f"Saved screenshot as {filename}")
    elif key == ord("t"):  # Toggle grayscale
        gray_mode = not gray_mode
        print(f"Grayscale mode: {'ON' if gray_mode else 'OFF'}")
    elif key == ord("b"):  # Toggle face blurring
        blur_faces = not blur_faces
        print(f"Face blurring: {'ON' if blur_faces else 'OFF'}")
    elif key == ord("r"):  # Start/stop recording
        is_recording = not is_recording
        if is_recording:
            print("Recording started...")
        else:
            if video_writer is not None:
                video_writer.release()
                video_writer = None
                recording_start_time = None
                print("Recording stopped and saved.")
    elif key == ord("g"):  # Toggle age/gender simulation
        age_gender_mode = not age_gender_mode
        print(f"Age/Gender mode: {'ON' if age_gender_mode else 'OFF'}")
    elif key == ord("e"):  # Toggle edge detection
        edge_detection = not edge_detection
        print(f"Edge detection: {'ON' if edge_detection else 'OFF'}")
    elif key == ord("m"):  # Toggle motion detection
        motion_detection = not motion_detection
        print(f"Motion detection: {'ON' if motion_detection else 'OFF'}")
    elif key == ord("n"):  # Toggle night mode
        night_mode = not night_mode
        print(f"Night mode: {'ON' if night_mode else 'OFF'}")
    elif key == ord("f"):  # Toggle face tracking
        face_tracking = not face_tracking
        print(f"Face tracking: {'ON' if face_tracking else 'OFF'}")
    elif key == ord("x"):  # Toggle emotion simulation
        emotion_mode = not emotion_mode
        print(f"Emotion mode: {'ON' if emotion_mode else 'OFF'}")
    elif key == ord("l"):  # Toggle landmarks
        landmark_mode = not landmark_mode
        print(f"Landmark mode: {'ON' if landmark_mode else 'OFF'}")
    elif key == ord("z"):  # Toggle zoom
        zoom_mode = not zoom_mode
        print(f"Zoom mode: {'ON' if zoom_mode else 'OFF'}")
    elif key == ord("+"):  # Increase zoom
        if zoom_mode:
            zoom_factor = min(max_zoom, zoom_factor + 0.1)
            print(f"Zoom factor: {zoom_factor:.1f}")
    elif key == ord("-"):  # Decrease zoom
        if zoom_mode:
            zoom_factor = max(1.0, zoom_factor - 0.1)
            print(f"Zoom factor: {zoom_factor:.1f}")
    elif key == ord("1"):  # Mirror mode toggle
        mirror_mode = not mirror_mode
        print(f"Mirror mode: {'ON' if mirror_mode else 'OFF'}")
    elif key == ord("2"):  # Increase night brightness
        night_brightness = min(100, night_brightness + 5)
        print(f"Night brightness: {night_brightness}")
    elif key == ord("3"):  # Decrease night brightness
        night_brightness = max(-100, night_brightness - 5)
        print(f"Night brightness: {night_brightness}")
    elif key == ord("4"):  # Increase night contrast
        night_contrast = min(100, night_contrast + 5)
        print(f"Night contrast: {night_contrast}")
    elif key == ord("5"):  # Decrease night contrast
        night_contrast = max(-100, night_contrast - 5)
        print(f"Night contrast: {night_contrast}")
    elif key == ord("h"):  # Help menu
        print("\n=== Key Controls ===")
        print("a - Exit")
        print("s - Save screenshot")
        print("t - Toggle grayscale")
        print("b - Toggle face blurring")
        print("r - Start/stop recording")
        print("g - Toggle age/gender simulation")
        print("e - Toggle edge detection")
        print("m - Toggle motion detection")
        print("n - Toggle night mode")
        print("f - Toggle face tracking")
        print("x - Toggle emotion simulation")
        print("l - Toggle landmark simulation")
        print("z - Toggle zoom")
        print("+ - Increase zoom")
        print("- - Decrease zoom")
        print("1 - Toggle mirror mode")
        print("2/3 - Adjust night brightness")
        print("4/5 - Adjust night contrast")
        print("h - Show this help menu\n")

# Release resources
if video_writer is not None:
    video_writer.release()
video_cap.release()
cv2.destroyAllWindows()

# Print summary statistics
print("\n=== Session Summary ===")
print(f"Total runtime: {duration_str}")
print(f"Maximum faces detected at once: {max(prev_face_count, face_count)}")
print(f"Total unique faces detected: {next_face_id - 1 if face_tracking else total_faces_detected}")
print(f"Average FPS: {avg_fps:.1f}")