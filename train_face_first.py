import cv2
import os
import numpy as np
import pickle
import time

print("🎯 FIRST TIME SETUP: Training Your Face")
print("This will train the system to recognize you as OWNER")
print("=" * 50)

# Create necessary folders
if not os.path.exists('faces_dataset'):
    os.makedirs('faces_dataset')
if not os.path.exists('faces_dataset/owner'):
    os.makedirs('faces_dataset/owner')

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

print("📷 Starting camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera! Check if it's being used by another app.")
    exit()

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("✅ Camera ready! Look straight at the camera")
print("🔄 Capturing 30 images (this will take about 30 seconds)")
print("💡 Make sure your face is well-lit and visible")

face_samples = []
labels = []
count = 0
label_id = 0  # 0 = owner

while count < 30:  # Capture 30 images instead of 100 for faster testing
    ret, frame = cap.read()
    if not ret:
        print("❌ Can't receive frame. Trying again...")
        continue
    
    # Flip frame so it's like mirror
    frame = cv2.flip(frame, 1)
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Only use clear faces
        if w > 100 and h > 100:
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            
            face_samples.append(face_roi)
            labels.append(label_id)
            
            # Save sample image
            cv2.imwrite(f'faces_dataset/owner/face_{count}.jpg', face_roi)
            
            count += 1
            print(f"📸 Captured image {count}/30")
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Training: {count}/30", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show progress
    cv2.putText(frame, f"FACE TRAINING: {count}/30", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Look straight at camera", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('Face Training - Press Q to cancel', frame)
    
    # Wait a bit between captures
    time.sleep(0.5)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("❌ Training cancelled by user")
        break

cap.release()
cv2.destroyAllWindows()

if len(face_samples) > 0:
    print("🔄 Training the model...")
    
    # Train the model
    recognizer.train(face_samples, np.array(labels))
    recognizer.save('face_model.yml')
    
    # Save labels
    labels_dict = {'owner': 0}
    with open('labels.pickle', 'wb') as f:
        pickle.dump(labels_dict, f)
    
    print("✅ TRAINING COMPLETED!")
    print(f"✅ Trained with {len(face_samples)} face images")
    print("✅ Model saved as 'face_model.yml'")
    print("✅ You can now run the main security system!")
    print("\n🎯 Next step: Run 'python run_system.py'")
else:
    print("❌ No faces captured! Please check:")
    print("   - Camera is working")
    print("   - Face is visible and well-lit")
    print("   - No other apps using camera")