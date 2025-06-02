import cv2
import numpy as np
import os
from PIL import Image

class FaceRecognizer:
    def __init__(self, database):
        """
        Initialize face recognizer with database connection.
        """
        self.database = database
        # Use relative path for the cascade file
        cascade_path = os.path.join(os.path.dirname(__file__), 'data', 'haarcascade_frontalface_default.xml')
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Initialize the LBPH face recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer.create()
        self.known_face_encodings = {}
        self.load_encodings()
        
    def load_encodings(self):
        """Load all face encodings from the database."""
        self.known_face_encodings = self.database.get_all_encodings()
        
        # Try to load the trained model if it exists
        if os.path.exists('face_recognizer_model.yml'):
            try:
                self.recognizer.read('face_recognizer_model.yml')
            except Exception as e:
                print(f"Error loading face recognizer model: {e}")
        
    def detect_faces(self, frame):
        """
        Detect faces in a frame using Haar cascade.
        
        Returns:
            List of (x, y, w, h) face locations
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces, gray
    
    def encode_face(self, face_image):
        """
        Generate face encoding for a given face image.
        
        Args:
            face_image: RGB image containing a face
            
        Returns:
            Face image in grayscale
        """
        # Convert to grayscale
        if len(face_image.shape) == 3:
            gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = face_image
            
        # Resize for consistency
        gray_face = cv2.resize(gray_face, (100, 100))
        
        return gray_face
    
    def recognize_face(self, face_image):
        """
        Compare face with known faces.
        
        Args:
            face_image: Grayscale face image
            
        Returns:
            User ID if match found, None otherwise
        """
        if not self.known_face_encodings:
            return None
            
        if not os.path.exists('face_recognizer_model.yml'):
            return None
            
        # Resize for consistency
        face_image = cv2.resize(face_image, (100, 100))
        
        # Predict
        try:
            label, confidence = self.recognizer.predict(face_image)
            
            # Lower confidence is better in LBPH
            if confidence < 70:  # Threshold for recognition
                return label
        except:
            pass
            
        return None
    
    def process_frame(self, frame, mark_attendance=True, draw_box=True):
        """
        Process a video frame to detect and recognize faces.
        
        Args:
            frame: BGR frame from camera
            mark_attendance: Whether to mark attendance for recognized faces
            draw_box: Whether to draw bounding boxes on the frame
            
        Returns:
            Processed frame and list of recognized user IDs
        """
        recognized_users = []
        
        # Detect faces
        faces, gray = self.detect_faces(frame)
        
        # Process each face
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Recognize face
            user_id = self.recognize_face(face_roi)
            
            if user_id:
                recognized_users.append(user_id)
                
                # Mark attendance
                if mark_attendance:
                    self.database.mark_attendance(user_id)
                
                # Get user name
                user = self.database.get_user(user_id)
                name = user[1] if user else "Unknown"
                
                # Draw box and name
                if draw_box:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y+h-35), (x+w, y+h), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, name, (x+6, y+h-6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            else:
                # Draw box for unknown face
                if draw_box:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.rectangle(frame, (x, y+h-35), (x+w, y+h), (0, 0, 255), cv2.FILLED)
                    cv2.putText(frame, "Unknown", (x+6, y+h-6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        
        return frame, recognized_users
    
    def capture_face(self, frame):
        """
        Capture a face from a frame for enrollment.
        
        Returns:
            Face image if a face is detected, None otherwise
        """
        # Detect faces
        faces, gray = self.detect_faces(frame)
        
        if len(faces) == 0:
            return None
        
        # Use the first face detected
        x, y, w, h = faces[0]
        
        # Extract face image
        face_image = frame[y:y+h, x:x+w]
        
        return face_image
    
    def save_face_image(self, face_image, user_name, directory="face_images"):
        """
        Save a face image to disk.
        
        Args:
            face_image: RGB face image
            user_name: Name of the user
            directory: Directory to save images
            
        Returns:
            Path to the saved image
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Generate filename
        filename = f"{user_name}_{np.random.randint(10000)}.jpg"
        file_path = os.path.join(directory, filename)
        
        # Convert numpy array to PIL Image and save
        pil_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        pil_image.save(file_path)
        
        return file_path
    
    def train_recognizer(self):
        """Train the face recognizer with all faces in the database."""
        if not self.known_face_encodings:
            return False
            
        faces = []
        labels = []
        
        for user_id, face_encoding in self.known_face_encodings.items():
            # Convert the encoding back to a face image (grayscale)
            face_image = face_encoding
            faces.append(face_image)
            labels.append(user_id)
        
        if not faces:
            return False
            
        # Train the recognizer
        self.recognizer.train(faces, np.array(labels))
        
        # Save the model
        self.recognizer.save('face_recognizer_model.yml')
        
        return True
    
    def enroll_user(self, frame, name, role):
        """
        Enroll a new user from a video frame.
        
        Args:
            frame: BGR frame from camera
            name: User name
            role: User role
            
        Returns:
            User ID if enrollment successful, None otherwise
        """
        # Capture face
        face_image = self.capture_face(frame)
        
        if face_image is None:
            return None
        
        # Save face image
        image_path = self.save_face_image(face_image, name)
        
        # Generate face encoding (grayscale face image)
        gray_face = self.encode_face(face_image)
        
        # Add user to database
        user_id = self.database.add_user(name, role, gray_face, image_path)
        
        # Update known face encodings
        self.known_face_encodings[user_id] = gray_face
        
        # Train the recognizer
        self.train_recognizer()
        
        return user_id
    
    def check_liveness(self, frame):
        """
        Basic liveness detection.
        This is a simplified implementation and not very robust.
        
        Returns:
            True if the face is likely real, False otherwise
        """
        # This would be a placeholder for a more sophisticated liveness detection
        # For now, we'll just return True
        return True
