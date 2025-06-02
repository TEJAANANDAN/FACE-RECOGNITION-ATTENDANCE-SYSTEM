import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
import os

from database import Database
from face_recognition_module import FaceRecognizer
from ui.dashboard import Dashboard
from ui.enrollment import EnrollmentPage
from ui.attendance_history import AttendanceHistory

class FaceAttendanceSystem(tk.Tk):
    def __init__(self):
        super().__init__()
        
        # Configure window
        self.title("Face Recognition Attendance System")
        self.geometry("1200x700")
        self.minsize(1000, 600)
        
        # Initialize database
        try:
            self.database = Database()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize database: {str(e)}")
            self.quit()
        
        try:
            self.face_recognizer = FaceRecognizer(self.database)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize face recognizer: {str(e)}")
            self.quit()
        
        # Create UI
        self.create_widgets()
        
        # Bind closing event
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_widgets(self):
        """Create main application widgets."""
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.dashboard = Dashboard(self.notebook, self.database, self.face_recognizer)
        self.notebook.add(self.dashboard, text="Dashboard")
        
        self.enrollment = EnrollmentPage(self.notebook, self.database, self.face_recognizer)
        self.notebook.add(self.enrollment, text="User Enrollment")
        
        self.attendance_history = AttendanceHistory(self.notebook, self.database)
        self.notebook.add(self.attendance_history, text="Attendance History")
    
    def on_closing(self):
        """Handle application closing."""
        self.dashboard.on_closing()
        self.enrollment.on_closing()
        self.database.close()
        self.destroy()

if __name__ == "__main__":
    app = FaceAttendanceSystem()

    # Start the application
    app.mainloop()
