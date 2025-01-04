import warnings
warnings.filterwarnings("ignore")
from typing import Optional, List, Union
import sys
import os
import datetime
import threading
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QPushButton, QLabel, QStackedWidget, QProgressBar)
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QPalette
import sounddevice as sd
import numpy as np
import soundfile as sf
from backend_processing import process_audio_file, compress_audio_files

class SignalManager(QObject):
    """
    Handles all the signal communication between different parts of the app.
    Makes it easier to keep track of what's happening where.
    """
    recording_status = pyqtSignal(bool)  # True when recording starts, False when it stops
    processing_started = pyqtSignal()    # Emitted when audio processing begins
    processing_complete = pyqtSignal(str, str)  # Emitted with question and answer text
    error_occurred = pyqtSignal(str)     # Emitted when something goes wrong

class RecordScreen(QWidget):
    """
    Screen where users can record their questions. Shows a button to start/stop
    recording and displays the current status.
    """
    def __init__(self, signal_manager: SignalManager) -> None:
        super().__init__()
        self.signal_manager = signal_manager
        self.is_recording: bool = False
        self.audio_data: Optional[List[np.ndarray]] = None
        self.sample_rate: int = 44100
        
        # Set up the UI
        layout = QVBoxLayout()
        
        self.status_label = QLabel("Press button or Tap Enter key to start recording")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        
        layout.addWidget(self.status_label)
        layout.addWidget(self.record_button)
        self.setLayout(layout)
        
    def toggle_recording(self) -> None:
        """Switches between recording and not recording states"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self) -> None:
        """Sets up and starts the audio recording stream"""
        self.is_recording = True
        self.audio_data = []
        self.record_button.setText("Stop Recording")
        self.status_label.setText("Recording...")
        self.signal_manager.recording_status.emit(True)
        
        # Callback function that gets audio data from the mic
        def callback(indata: np.ndarray, frames: int, time: float, status: Optional[str]) -> None:
            if status:
                print(status)  # Print any stream errors
            self.audio_data.append(indata.copy())
            
        # Set up audio input stream
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=callback
        )
        self.stream.start()
    
    def stop_recording(self) -> None:
        """
        Stops recording and kicks off audio processing in a background thread.
        Using a thread keeps the UI responsive while processing happens.
        """
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
            self.is_recording = False
            self.record_button.setText("Start Recording")
            self.status_label.setText("Processing...")
            
            # Combine all our audio chunks
            audio_data = np.concatenate(self.audio_data)
            self.signal_manager.recording_status.emit(False)
            
            # Process audio in background
            processing_thread = threading.Thread(
                target=self.process_audio,
                args=(audio_data,)
            )
            processing_thread.start()
    
    def process_audio(self, audio_data: np.ndarray) -> None:
        """
        Handles all the audio processing steps:
        1. Creates output folders
        2. Saves the raw audio
        3. Processes it through our backend
        4. Compresses the results
        """
        try:
            self.signal_manager.processing_started.emit()
            
            # Make folders to store everything
            output_directory = "Questions"
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            question_folder = os.path.join(output_directory, current_time)
            clean_folder = os.path.join(question_folder, "clean")
            noisy_folder = os.path.join(question_folder, "noisy")
            os.makedirs(clean_folder, exist_ok=True)
            os.makedirs(noisy_folder, exist_ok=True)
            
            # Save the original audio
            original_path = os.path.join(noisy_folder, "original.wav")
            sf.write(original_path, audio_data, self.sample_rate)
            
            # Run it through our processing pipeline
            question_text, answer_text = process_audio_file(original_path)
            compress_audio_files(question_folder)
            
            self.signal_manager.processing_complete.emit(question_text, answer_text)
            
        except Exception as e:
            self.signal_manager.error_occurred.emit(str(e))

class ProcessingScreen(QWidget):
    """
    Simple loading screen shown while audio is being processed.
    Just shows a progress bar and status message.
    """
    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout()
        
        self.status_label = QLabel("Processing your audio...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Makes the progress bar do the bouncy thing
        
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)

class ResultScreen(QWidget):
    """
    Shows the transcribed question and AI's answer.
    Has a button to start over with a new question.
    """
    def __init__(self, signal_manager: SignalManager) -> None:
        super().__init__()
        self.signal_manager = signal_manager
        
        layout = QVBoxLayout()
        
        self.result_label = QLabel()
        self.result_label.setWordWrap(True)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.new_question_button = QPushButton("Ask Another Question")
        self.new_question_button.clicked.connect(self.return_to_record)
        
        layout.addWidget(self.result_label)
        layout.addWidget(self.new_question_button)
        self.setLayout(layout)
    
    def set_result(self, question: str, answer: str) -> None:
        """Updates the display with new Q&A"""
        self.result_label.setText(f"Q: {question}\n\nA: {answer}")
    
    def return_to_record(self) -> None:
        """Takes us back to recording screen"""
        self.signal_manager.recording_status.emit(True)

class MainWindow(QMainWindow):
    """
    Main application window that manages all our different screens.
    Uses a stacked widget to switch between screens as needed.
    """
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Audio Q&A")
        self.setMinimumSize(400, 300)
        
        # Set up signal handling
        self.signal_manager = SignalManager()
        
        # Create all our screens
        self.stacked_widget = QStackedWidget()
        self.record_screen = RecordScreen(self.signal_manager)
        self.processing_screen = ProcessingScreen()
        self.result_screen = ResultScreen(self.signal_manager)
        
        # Add them to the stack
        self.stacked_widget.addWidget(self.record_screen)
        self.stacked_widget.addWidget(self.processing_screen)
        self.stacked_widget.addWidget(self.result_screen)
        
        # Hook up all our signals
        self.signal_manager.recording_status.connect(self.handle_recording_status)
        self.signal_manager.processing_started.connect(self.show_processing)
        self.signal_manager.processing_complete.connect(self.show_result)
        self.signal_manager.error_occurred.connect(self.handle_error)
        
        self.setCentralWidget(self.stacked_widget)
    
    def handle_recording_status(self, is_recording: bool) -> None:
        """Switches to record screen when recording starts"""
        if is_recording:
            self.stacked_widget.setCurrentWidget(self.record_screen)
    
    def show_processing(self) -> None:
        """Shows the loading screen"""
        self.stacked_widget.setCurrentWidget(self.processing_screen)
    
    def show_result(self, question: str, answer: str) -> None:
        """Shows the results screen with processed Q&A"""
        self.result_screen.set_result(question, answer)
        self.stacked_widget.setCurrentWidget(self.result_screen)
    
    def handle_error(self, error_message: str) -> None:
        """Basic error handling - just prints and goes back to start"""
        print(f"Error: {error_message}")
        self.stacked_widget.setCurrentWidget(self.record_screen)
    
    def keyPressEvent(self, event: Qt.Key) -> None:
        """Lets users start/stop recording with Enter key"""
        if event.key() == Qt.Key.Key_Return:
            if self.stacked_widget.currentWidget() == self.record_screen:
                self.record_screen.toggle_recording()

def main() -> None:
    app = QApplication(sys.argv)
    
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, Qt.GlobalColor.black)
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    app.setPalette(palette)
    
    app.setApplicationName("Audio Q&A")
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()