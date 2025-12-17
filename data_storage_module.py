import json
import csv
import os
import sqlite3
from datetime import datetime
import logging
import pickle
import cv2

class DataStorage:
    def __init__(self):
        self.logger = self.setup_logging()
        self.db_file = "security_system.db"
        self.events_file = "security_events.json"
        self.alerts_file = "alerts_log.csv"
        
        self.setup_database()
        self.setup_storage()
    
    def setup_logging(self):
        """Setup logging for data storage"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('data_storage.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def setup_database(self):
        """Initialize SQLite database"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            # Create events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS security_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    person_name TEXT,
                    confidence REAL,
                    camera_location TEXT,
                    additional_info TEXT
                )
            ''')
            
            # Create alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT,
                    description TEXT,
                    action_taken TEXT,
                    resolved BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # Create face logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS face_recognition_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    recognized_name TEXT,
                    confidence REAL,
                    image_path TEXT,
                    camera_id INTEGER
                )
            ''')
            
            conn.commit()
            conn.close()
            self.logger.info("Database setup completed")
            
        except Exception as e:
            self.logger.error(f"Error setting up database: {e}")
    
    def setup_storage(self):
        """Setup file-based storage"""
        try:
            # Create directories for storing evidence
            directories = ['evidence/images', 'evidence/videos', 'evidence/gesture_queue/incoming', 'evidence/gesture_queue/processed', 'evidence/gesture_queue/results', 'logs']
            for directory in directories:
                if not os.path.exists(directory):
                    os.makedirs(directory)
            
            # Initialize CSV file with headers if it doesn't exist
            if not os.path.exists(self.alerts_file):
                with open(self.alerts_file, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Timestamp', 'Alert Type', 'Severity', 'Description', 'Location', 'Action Taken'])
            
            self.logger.info("File storage setup completed")
            
        except Exception as e:
            self.logger.error(f"Error setting up file storage: {e}")
    
    def log_security_event(self, event_type, person_name=None, confidence=0, location="front_door", info=""):
        """Log security events to database and JSON"""
        try:
            timestamp = datetime.now().isoformat()
            
            # Database logging
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO security_events (timestamp, event_type, person_name, confidence, camera_location, additional_info)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (timestamp, event_type, person_name, confidence, location, info))
            
            conn.commit()
            conn.close()
            
            # JSON logging
            event_data = {
                'timestamp': timestamp,
                'event_type': event_type,
                'person_name': person_name,
                'confidence': confidence,
                'location': location,
                'additional_info': info
            }
            
            # Append to JSON file
            events = []
            if os.path.exists(self.events_file):
                with open(self.events_file, 'r') as f:
                    try:
                        events = json.load(f)
                    except json.JSONDecodeError:
                        events = []
            
            events.append(event_data)
            
            with open(self.events_file, 'w') as f:
                json.dump(events, f, indent=2)
            
            self.logger.info(f"Security event logged: {event_type} - {person_name}")
            
        except Exception as e:
            self.logger.error(f"Error logging security event: {e}")
    
    def log_alert(self, alert_type, severity, description, action_taken=""):
        """Log alerts to database and CSV"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Database logging
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO alerts (timestamp, alert_type, severity, description, action_taken)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, alert_type, severity, description, action_taken))
            
            conn.commit()
            conn.close()
            
            # CSV logging
            with open(self.alerts_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, alert_type, severity, description, "front_door", action_taken])
            
            self.logger.warning(f"Alert logged: {alert_type} - {severity} - {description}")
            
        except Exception as e:
            self.logger.error(f"Error logging alert: {e}")
    
    def save_evidence_image(self, frame, event_type, person_name="unknown"):
        """Save evidence images with timestamp"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evidence/images/{event_type}_{person_name}_{timestamp}.jpg"
            
            # Create thumbnail
            thumbnail = cv2.resize(frame, (320, 240))
            
            cv2.imwrite(filename, frame)
            cv2.imwrite(filename.replace(".jpg", "_thumb.jpg"), thumbnail)
            
            # Log in database
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO face_recognition_logs (timestamp, recognized_name, confidence, image_path, camera_id)
                VALUES (?, ?, ?, ?, ?)
            ''', (datetime.now().isoformat(), person_name, 0, filename, 1))
            
            conn.commit()
            conn.close()
            
            return filename
            
        except Exception as e:
            self.logger.error(f"Error saving evidence image: {e}")
            return None

    def save_evidence_clip(self, frames, event_type, person_name="unknown", fps=20):
        """Save a short video clip (AVI) as evidence.

        Args:
            frames: list of BGR numpy arrays
            event_type: short event name to include in filename
            person_name: label for filename
            fps: frames per second for the output video
        Returns:
            filename (str) or None
        """
        try:
            if not frames:
                self.logger.error("No frames provided to save_evidence_clip")
                return None

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evidence/videos/{event_type}_{person_name}_{timestamp}.avi"

            h, w = frames[0].shape[:2]
            # Use MJPG codec for wide compatibility
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(filename, fourcc, fps, (w, h))

            for f in frames:
                # Ensure frame has same size
                if f.shape[0] != h or f.shape[1] != w:
                    f = cv2.resize(f, (w, h))
                out.write(f)

            out.release()

            # Also save a thumbnail image
            thumb = cv2.resize(frames[len(frames)//2], (320, 240))
            thumb_name = filename.replace('.avi', '_thumb.jpg')
            cv2.imwrite(thumb_name, thumb)

            # Log as security event
            self.log_security_event(f"VIDEO_{event_type}", person_name, 0, "front_door", f"Saved clip: {filename}")

            self.logger.info(f"Saved evidence clip: {filename}")
            return filename

        except Exception as e:
            self.logger.error(f"Error saving evidence clip: {e}")
            return None

    def queue_clip_for_gesture(self, clip_path, bbox=None):
        """Copy an existing clip into the gesture worker input queue and write metadata.

        Args:
            clip_path: path to an existing video clip
            bbox: optional face bbox tuple (x,y,w,h) to include as metadata

        Returns:
            queue_path (str) or None
        """
        # Validate
        if not os.path.exists(clip_path):
            self.logger.error(f"Clip to queue not found: {clip_path}")
            return None

        try:
            import shutil, uuid, json
            uid = uuid.uuid4().hex
            dest = f"evidence/gesture_queue/incoming/gesture_{uid}.avi"
            shutil.copy2(clip_path, dest)

            # Write metadata file with same uid
            # Sanitize bbox and other data to ensure JSON serializable types
            safe_bbox = None
            if bbox is not None:
                try:
                    x, y, w, h = bbox
                    safe_bbox = [int(x), int(y), int(w), int(h)]
                except Exception:
                    # If bbox is not a simple iterable, ignore
                    safe_bbox = None

            meta = {
                'clip': dest,
                'bbox': safe_bbox,
                'timestamp': datetime.now().isoformat()
            }
            meta_path = dest.replace('.avi', '.json')

            # Use the sanitizer to remove numpy / non-JSON types
            try:
                safe_meta = self._sanitize_for_json(meta)
            except Exception:
                safe_meta = meta

            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(safe_meta, f, indent=2)

            self.logger.info(f"Queued clip for gesture analysis: {dest}")
            return dest
        except Exception as e:
            self.logger.error(f"Error queueing clip for gesture: {e}")
            return None

    def _sanitize_for_json(self, obj):
        """Recursively convert numpy and other non-JSON-safe types to native Python types.

        This will convert numpy integers/floats to int/float, numpy arrays to lists,
        and ensure nested dict/list structures are sanitized.
        """
        try:
            import numpy as np
        except Exception:
            np = None

        # Primitive safe types
        if obj is None or isinstance(obj, (str, bool, int, float)):
            return obj

        # Numpy scalar types
        if np is not None:
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.ndarray,)):
                return [self._sanitize_for_json(x) for x in obj.tolist()]

        # Dict - sanitize keys and values
        if isinstance(obj, dict):
            sanitized = {}
            for k, v in obj.items():
                # Ensure key is a string
                key = str(k)
                sanitized[key] = self._sanitize_for_json(v)
            return sanitized

        # List / tuple / set
        if isinstance(obj, (list, tuple, set)):
            return [self._sanitize_for_json(x) for x in obj]

        # Bytes -> decode
        if isinstance(obj, (bytes, bytearray)):
            try:
                return obj.decode('utf-8')
            except Exception:
                return str(obj)

        # Fallback: try to cast to primitive
        try:
            return int(obj)
        except Exception:
            try:
                return float(obj)
            except Exception:
                return str(obj)

    def get_recent_events(self, limit=10):
        """Retrieve recent security events"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM security_events 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            events = cursor.fetchall()
            conn.close()
            
            return events
            
        except Exception as e:
            self.logger.error(f"Error retrieving events: {e}")
            return []
    
    def get_alerts_by_severity(self, severity):
        """Get alerts by severity level"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM alerts 
                WHERE severity = ? AND resolved = FALSE
                ORDER BY timestamp DESC
            ''', (severity,))
            
            alerts = cursor.fetchall()
            conn.close()
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error retrieving alerts: {e}")
            return []
    
    def export_data(self, start_date, end_date, export_format='json'):
        """Export data for specified date range"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM security_events 
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            ''', (start_date, end_date))
            
            events = cursor.fetchall()
            conn.close()
            
            if export_format == 'json':
                export_file = f"export_events_{start_date}_{end_date}.json"
                with open(export_file, 'w') as f:
                    json.dump(events, f, indent=2)
            elif export_format == 'csv':
                export_file = f"export_events_{start_date}_{end_date}.csv"
                with open(export_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['ID', 'Timestamp', 'Event Type', 'Person', 'Confidence', 'Location', 'Info'])
                    writer.writerows(events)
            
            self.logger.info(f"Data exported to {export_file}")
            return export_file
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            return None