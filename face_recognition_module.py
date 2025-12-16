import cv2
import numpy as np
import pickle
import os
import time
from datetime import datetime
import logging

class FaceRecognition:
    def __init__(self):
        self.logger = self.setup_logging()
        
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Face recognition model
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.setThreshold(70)  # Lower threshold = stricter recognition
        
        # Data storage
        self.faces_dir = "faces_dataset"
        self.model_file = "face_model.yml"
        self.labels_file = "labels.pickle"
        
        # Labels mapping
        self.label_ids = {}
        self.current_id = 0
        self.labels = {}
        
        self.setup_directories()
        self.load_model()
    
    def setup_logging(self):
        """Setup logging for face recognition"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('face_recognition.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def setup_directories(self):
        """Create necessary directories"""
        try:
            if not os.path.exists(self.faces_dir):
                os.makedirs(self.faces_dir)
                self.logger.info(f"Created directory: {self.faces_dir}")
        except Exception as e:
            self.logger.error(f"Error creating directories: {e}")
    
    def load_model(self):
        """Load existing face recognition model"""
        try:
            if os.path.exists(self.model_file) and os.path.exists(self.labels_file):
                self.recognizer.read(self.model_file)
                with open(self.labels_file, 'rb') as f:
                    self.label_ids = pickle.load(f)
                self.labels = {v: k for k, v in self.label_ids.items()}
                self.current_id = max(self.label_ids.values()) + 1 if self.label_ids else 0
                self.logger.info("Face recognition model loaded successfully")
                self.logger.info(f"Known faces: {list(self.label_ids.keys())}")
            else:
                self.logger.warning("No trained model found. Please train with owner's face.")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
    
    def detect_faces(self, frame):
        """Detect faces in frame with enhanced detection"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with multiple parameters for better detection
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Enhanced face verification with eye detection
            verified_faces = []
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                
                # Verify it's actually a face by detecting eyes
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                
                # If eyes detected or face is large enough, consider it valid
                if len(eyes) >= 1 or w > 100:
                    verified_faces.append((x, y, w, h))
            
            return verified_faces, gray
        
        except Exception as e:
            self.logger.error(f"Error in face detection: {e}")
            return [], None
    
    def recognize_face(self, frame, draw_boxes=True):
        """Recognize faces in the frame"""
        try:
            faces, gray = self.detect_faces(frame)
            recognition_results = []
            owner_detected = False
            confidence_level = 0
            
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                
                # Resize for consistent recognition
                roi_gray = cv2.resize(roi_gray, (200, 200))
                
                # Predict face
                label_id, confidence = self.recognizer.predict(roi_gray)
                
                # Lower confidence = better match
                if confidence < 60:  # Strict threshold
                    name = self.labels.get(label_id, "Unknown")
                    if name.lower() == "owner":
                        owner_detected = True
                        confidence_level = 100 - confidence
                        color = (0, 255, 0)  # Green
                        status = f"OWNER ({confidence_level:.1f}%)"
                    else:
                        color = (255, 255, 0)  # Yellow for known but not owner
                        status = f"Known: {name}"
                else:
                    color = (0, 0, 255)  # Red for unknown
                    status = "UNKNOWN"
                
                recognition_results.append({
                    'bbox': (x, y, w, h),
                    'name': status,
                    'color': color,
                    'confidence': confidence
                })
                
                if draw_boxes:
                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Draw label background
                    label_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x, y-label_size[1]-10), (x+label_size[0], y), color, -1)
                    
                    # Draw label text
                    cv2.putText(frame, status, (x, y-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return frame, owner_detected, recognition_results, confidence_level
        
        except Exception as e:
            self.logger.error(f"Error in face recognition: {e}")
            return frame, False, [], 0
    
    def train_new_face(self, name="owner", capture_count=100):
        """Train the system with a new face"""
        try:
            self.logger.info(f"Starting face training for: {name}")
            
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self.logger.error("Cannot open camera")
                return False
            
            face_samples = []
            labels = []
            count = 0
            
            # Create person-specific directory
            person_dir = os.path.join(self.faces_dir, name)
            if not os.path.exists(person_dir):
                os.makedirs(person_dir)
            
            self.logger.info(f"Look at the camera and move your head slightly. Capturing {capture_count} images...")
            
            while count < capture_count:
                ret, frame = cap.read()
                if not ret:
                    self.logger.error("Can't receive frame")
                    break
                
                faces, gray = self.detect_faces(frame)
                
                for (x, y, w, h) in faces:
                    # Only use clear, well-positioned faces
                    if w > 100 and h > 100:  # Minimum face size
                        roi_gray = gray[y:y+h, x:x+w]
                        roi_gray = cv2.resize(roi_gray, (200, 200))
                        
                        face_samples.append(roi_gray)
                        
                        # Assign label ID
                        if name not in self.label_ids:
                            self.label_ids[name] = self.current_id
                            self.current_id += 1
                        
                        labels.append(self.label_ids[name])
                        
                        # Save sample image
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        cv2.imwrite(os.path.join(person_dir, f"{name}_{timestamp}.jpg"), roi_gray)
                        
                        count += 1
                        
                        # Display progress
                        cv2.putText(frame, f"Training: {count}/{capture_count}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                cv2.imshow('Face Training - Press Q to cancel', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.logger.info("Training cancelled by user")
                    cap.release()
                    cv2.destroyAllWindows()
                    return False
            
            cap.release()
            cv2.destroyAllWindows()
            
            if len(face_samples) > 0:
                # Train the model
                self.recognizer.train(face_samples, np.array(labels))
                self.recognizer.save(self.model_file)
                
                # Save labels
                with open(self.labels_file, 'wb') as f:
                    pickle.dump(self.label_ids, f)
                
                self.labels = {v: k for k, v in self.label_ids.items()}
                
                self.logger.info(f"Training completed! Trained with {len(face_samples)} images for {name}")
                return True
            else:
                self.logger.error("No faces captured for training")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during face training: {e}")
            return False
    
    def get_known_faces(self):
        """Return list of known faces"""
        return list(self.label_ids.keys())