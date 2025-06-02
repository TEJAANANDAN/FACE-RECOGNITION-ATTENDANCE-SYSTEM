import sqlite3
import os
import pickle
import datetime
import pandas as pd
import numpy as np

class Database:
    def __init__(self, db_path="attendance.db"):
        """Initialize database connection and create tables if they don't exist."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_tables()
    
    def _create_tables(self):
        """Create required tables if they don't exist."""
        # Users table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            role TEXT,
            face_encoding BLOB,
            image_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Attendance table
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            date TEXT,
            time TEXT,
            status TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        self.conn.commit()
    
    def add_user(self, name, role, face_encoding, image_path):
        """Add a new user to the database."""
        face_encoding_blob = pickle.dumps(face_encoding)
        self.cursor.execute(
            "INSERT INTO users (name, role, face_encoding, image_path) VALUES (?, ?, ?, ?)",
            (name, role, face_encoding_blob, image_path)
        )
        self.conn.commit()
        return self.cursor.lastrowid
    
    def update_user(self, user_id, name=None, role=None, face_encoding=None, image_path=None):
        """Update user information."""
        updates = []
        params = []
        
        if name:
            updates.append("name = ?")
            params.append(name)
        if role:
            updates.append("role = ?")
            params.append(role)
        if face_encoding is not None:
            updates.append("face_encoding = ?")
            params.append(pickle.dumps(face_encoding))
        if image_path:
            updates.append("image_path = ?")
            params.append(image_path)
            
        if not updates:
            return False
            
        params.append(user_id)
        query = f"UPDATE users SET {', '.join(updates)} WHERE id = ?"
        self.cursor.execute(query, params)
        self.conn.commit()
        return True
    
    def delete_user(self, user_id):
        """Delete a user from the database."""
        self.cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        self.conn.commit()
        return self.cursor.rowcount > 0
    
    def get_all_users(self):
        """Get all users from the database."""
        self.cursor.execute("SELECT id, name, role, image_path, created_at FROM users")
        return self.cursor.fetchall()
    
    def get_user(self, user_id):
        """Get a specific user by ID."""
        self.cursor.execute("SELECT id, name, role, image_path, created_at FROM users WHERE id = ?", (user_id,))
        return self.cursor.fetchone()
    
    def get_user_encoding(self, user_id):
        """Get face encoding for a specific user."""
        self.cursor.execute("SELECT face_encoding FROM users WHERE id = ?", (user_id,))
        result = self.cursor.fetchone()
        if result and result[0]:
            return pickle.loads(result[0])
        return None
    
    def get_all_encodings(self):
        """Get all face encodings with corresponding user IDs."""
        self.cursor.execute("SELECT id, face_encoding FROM users WHERE face_encoding IS NOT NULL")
        results = self.cursor.fetchall()
        encodings = {}
        for user_id, encoding_blob in results:
            if encoding_blob:
                encodings[user_id] = pickle.loads(encoding_blob)
        return encodings
    
    def mark_attendance(self, user_id, status="present"):
        """Mark attendance for a user."""
        now = datetime.datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")
        
        # Check if attendance already marked for today
        self.cursor.execute(
            "SELECT id FROM attendance WHERE user_id = ? AND date = ?", 
            (user_id, date)
        )
        if self.cursor.fetchone():
            # Update existing attendance
            self.cursor.execute(
                "UPDATE attendance SET time = ?, status = ? WHERE user_id = ? AND date = ?",
                (time, status, user_id, date)
            )
        else:
            # Insert new attendance record
            self.cursor.execute(
                "INSERT INTO attendance (user_id, date, time, status) VALUES (?, ?, ?, ?)",
                (user_id, date, time, status)
            )
        
        self.conn.commit()
        return True
    
    def get_attendance(self, date=None, user_id=None):
        """Get attendance records with filters."""
        query = """
        SELECT a.id, a.user_id, u.name, a.date, a.time, a.status 
        FROM attendance a
        JOIN users u ON a.user_id = u.id
        """
        params = []
        
        conditions = []
        if date:
            conditions.append("a.date = ?")
            params.append(date)
        if user_id:
            conditions.append("a.user_id = ?")
            params.append(user_id)
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        query += " ORDER BY a.date DESC, a.time DESC"
        
        self.cursor.execute(query, params)
        return self.cursor.fetchall()
    
    def export_attendance(self, start_date=None, end_date=None, output_path="attendance_report.csv"):
        """Export attendance records to CSV."""
        query = """
        SELECT u.name, a.date, a.time, a.status 
        FROM attendance a
        JOIN users u ON a.user_id = u.id
        """
        params = []
        
        conditions = []
        if start_date:
            conditions.append("a.date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("a.date <= ?")
            params.append(end_date)
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        query += " ORDER BY a.date, u.name"
        
        self.cursor.execute(query, params)
        results = self.cursor.fetchall()
        
        df = pd.DataFrame(results, columns=["Name", "Date", "Time", "Status"])
        df.to_csv(output_path, index=False)
        return output_path
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
