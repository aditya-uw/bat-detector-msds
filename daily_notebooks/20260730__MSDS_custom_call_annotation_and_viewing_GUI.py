import numpy as np
import pandas as pd
import scipy
import dask.dataframe as dd

import soundfile as sf
import simpleaudio as sa
import librosa

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector

from pathlib import Path
import fsspec

import sys
import re

# These are the 6 locations studied using Audiomoths in 2022
SITE_NAMES = {
            'Central' : "Central Pond",
            'Foliage' : "Foliage",
            'Carp' : "Carp Pond",
            'Telephone' : "Telephone Field"
                }

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def plot_spectrogram(ax, row, audio_seg, location_df_orig, test_df, fs, duration, start, osn_file_path, row_index, nfft):
    """
    Plots a spectrogram for a given row of data with annotations.

    Parameters:
        ax: The matplotlib axis object.
        row: A row from the DataFrame containing details for the plot.
        location_df_kmeans_raw: DataFrame with detection details.
        fs: Sampling frequency (Hz).
        duration: Duration of the audio segment (seconds).
        start: Start time of the segment (seconds).
        FREQUENCY_COLOR_MAPPINGS: A dictionary mapping frequency groups to colors.
        osn_file_path: The path object for the file being processed.
    """
    # Plot the spectrogram
    ax.clear()
    ax.set_title(f'{osn_file_path.name}', fontsize=12)
    ax.specgram(audio_seg, Fs=fs, NFFT=nfft, cmap='jet', vmin=-120, vmax=-40)

    # Filter detection data for the current row
    file_df_orig = location_df_orig[location_df_orig['input_file'] == row['input_file']]
    plot_dets = file_df_orig[(file_df_orig['start_time'] >= start) & 
                             (file_df_orig['end_time'] <= (start + duration))]
    
    # Add rectangles for detections
    for _, det in plot_dets.iterrows():
        rect = patches.Rectangle(
            ((det['start_time'] - start), det['low_freq']),
            (det['end_time'] - det['start_time']),
            (det['high_freq'] - det['low_freq']),
            linewidth=1, edgecolor='yellow', facecolor='none', alpha=0.8
        )
        ax.add_patch(rect)

    # Configure plot labels and ticks
    ax.set_yticks(ticks=np.linspace(0, fs / 2, 6))
    ax.set_yticklabels(labels=np.linspace(0, fs / 2000, 6).astype('int'), fontsize=10)
    ax.set_ylabel("Frequency (kHz)", fontsize=10)

    ax.text(
        x=0.001, y=80000,
        s=f'Det {row_index+1} of {int(len(test_df))} added dets\nDetection score:{row["det_prob"]}\nSNR:{row["SNR"]}', 
        fontweight='bold', color='white', fontsize=10)
    
    ax.text(
        x=0.001, y=110000,
        s=f'{row["Site name"]} ({int(len(location_df_orig))} dets)', fontweight='bold', color='white', fontsize=10)

    ax.set_xticks(ticks=np.linspace(0, duration, 6))
    ax.set_xticklabels(labels=np.round(np.linspace(0, 0 + duration, 6, dtype=float), 2), fontsize=10)
    ax.set_xlabel(f"Time (s + {start:.2f}s)", fontsize=10)

def pitch_shift(audio_segment, original_sample_rate, pitch_factor, nfft_of_window):
    """
    Explicitly pitch-shift the audio signal by time-stretching and resampling.

    Parameters:
    - audio_segment (numpy.ndarray): The audio signal to process.
    - original_sample_rate (int): The sample rate of the audio signal.
    - pitch_factor (float): The pitch shift factor (>1 increases pitch, <1 decreases pitch).

    Returns:
    - pitch_shifted_audio (numpy.ndarray): The pitch-shifted audio signal.
    """
    rate = 2.0 ** (-float(pitch_factor) / 12)
    # Construct the short-term Fourier transform (STFT)
    stft = librosa.core.stft(audio_segment, n_fft=nfft_of_window)
    # Stretch by phase vocoding
    stft_stretch = librosa.core.phase_vocoder(stft,rate=rate,n_fft=nfft_of_window)
    # Predict the length of y_stretch
    len_stretch = int(round(audio_segment.shape[-1] / rate))
    # Invert the STFT
    time_stretched_signal = librosa.core.istft(stft_stretch, dtype=audio_segment.dtype, 
                                               length=len_stretch, n_fft=nfft_of_window)
    # Stretch in time, then resample
    y_shift = librosa.core.resample(time_stretched_signal,orig_sr=float(original_sample_rate) / rate,
                                    target_sr=original_sample_rate,res_type="soxr_hq",scale=False)
    # Crop to the same dimension as the input
    pitch_shifted_audio = librosa.util.fix_length(y_shift, size=audio_segment.shape[-1])

    return pitch_shifted_audio * (rate/2)

# GUI Application
class SpectrogramViewer:
    def __init__(self, root, dataframe, location_df_kmeans_raw):
        self.root = root
        self.dataframe = dataframe
        self.index = 0
        self.location_df_kmeans_raw = location_df_kmeans_raw
        self.cur_path = ''
        self.audio_file = ''
        self.fs = 250000
        self.processed_audio = None  # Store the processed audio data

        # Track annotations for each spectrogram
        self.annotations = {idx: "bat_call" for idx in dataframe.index}

        # Custom detections DataFrame
        self.custom_detections = pd.DataFrame(columns=["input_file", "start_time", "end_time", "low_freq", "high_freq", "category"])

        # Store the last drawn rectangle temporarily
        self.pending_rectangle = None

        # Drawing mode state
        self.drawing_mode = tk.BooleanVar(value=False)

        # Track the stack of drawn rectangles for undo functionality
        self.drawing_stack = []  # Holds references to the drawn rectangles

        # Current detection category for all new boxes in drawing mode
        self.current_category = tk.StringVar(value="custom_bat_call")

        # Main layout: Separate frames for organization
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill="both", expand=True)

        # Left frame for plot
        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.grid(row=0, column=0, rowspan=5, sticky="nsew")

        # Right frame for controls
        self.controls_frame = ttk.Frame(self.main_frame)
        self.controls_frame.grid(row=0, column=1, rowspan=5, sticky="nsew", padx=10)

        # Configure grid weights
        self.main_frame.columnconfigure(0, weight=3)  # Plot area
        self.main_frame.columnconfigure(1, weight=1)  # Controls
        self.main_frame.rowconfigure(0, weight=1)

        # Create the plot
        self.fig, self.ax = plt.subplots(figsize=(3, 3))  # Adjust plot size here
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)  # Place in the plot frame
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        self.canvas.draw()

        # Control widgets in the controls frame
        # Radiobuttons for annotations for the current detection
        self.radio_var = tk.StringVar(value="bat_call")
        self.bat_calls_button = ttk.Radiobutton(
            self.controls_frame, text="Bat Call", variable=self.radio_var, value="bat_call", command=self.set_annotation)
        self.feeding_buzzes_button = ttk.Radiobutton(
            self.controls_frame, text="Feeding Buzz", variable=self.radio_var, value="feeding_buzz", command=self.set_annotation)
        self.neither_button = ttk.Radiobutton(
            self.controls_frame, text="Neither", variable=self.radio_var, value="neither", command=self.set_annotation)
        self.bat_calls_button.grid(row=0, column=0, pady=5, sticky="w")
        self.feeding_buzzes_button.grid(row=1, column=0, pady=5, sticky="w")
        self.neither_button.grid(row=2, column=0, pady=5, sticky="w")

        # Navigation buttons
        self.left_button = ttk.Button(self.controls_frame, text="Previous", command=self.previous_row)
        self.left_button.grid(row=3, column=0, pady=5, sticky="w")
        self.right_button = ttk.Button(self.controls_frame, text="Next", command=self.next_row)
        self.right_button.grid(row=4, column=0, pady=5, sticky="w")

        # Playback controls
        self.play_button = ttk.Button(self.controls_frame, text="Play Audio", command=self.play_audio)
        self.play_button.grid(row=5, column=0, pady=5, sticky="w")
        self.save_audio_button = ttk.Button(self.controls_frame, text="Save Audio", command=self.save_audio_to_wav)
        self.save_audio_button.grid(row=6, column=0, pady=5, sticky="w")

        # NFFT slider and label
        self.nfft = tk.IntVar(value=256)
        self.nfft_label = ttk.Label(self.controls_frame, text="NFFT")
        self.nfft_label.grid(row=7, column=0, pady=5, sticky="w")
        self.nfft_slider = tk.Scale(self.controls_frame, from_=0, resolution=256, to=4096, variable=self.nfft,
            orient="horizontal", command=self.update_nfft, length=150, tickinterval=2048)
        self.nfft_slider.grid(row=7, column=0, pady=5, padx=(100, 0), sticky="w")
        self.nfft_slider.set(256)

        # DURATION slider and label
        self.duration = tk.DoubleVar(value=3.0)
        self.duration_label = ttk.Label(self.controls_frame, text="Duration")
        self.duration_label.grid(row=8, column=0, pady=5, sticky="w")
        self.duration_slider = tk.Scale(self.controls_frame, from_=0.1, resolution=0.1, to=5.0,
            variable=self.duration, orient="horizontal", command=self.update_duration, length=150, tickinterval=1.0)
        self.duration_slider.grid(row=8, column=0, pady=5, padx=(100, 0), sticky="w")
        self.duration_slider.set(3.0)

        # Octave shift slider
        self.pitch_shift = tk.IntVar(value=0.0)
        self.pitch_shift_label = ttk.Label(self.controls_frame, text="Octave Shift")
        self.pitch_shift_label.grid(row=9, column=0, pady=5, sticky="w")
        self.pitch_shift_slider = tk.Scale(self.controls_frame, from_=-3, to=3, resolution=1,
            variable=self.pitch_shift, orient="horizontal", length=150, tickinterval=1)
        self.pitch_shift_slider.grid(row=9, column=0, pady=5, padx=(100, 0), sticky="w")

        # Mode state variable
        self.is_drawing_mode = tk.BooleanVar(value=False)  # Default to "Browsing Mode"
        # Drawing mode toggle
        self.drawing_mode_toggle = ttk.Checkbutton(
            self.controls_frame, text="Drawing Mode", variable=self.is_drawing_mode, command=self.toggle_mode)
        self.drawing_mode_toggle.grid(row=11, column=0, pady=5, sticky="w")

        # Undo button
        self.undo_button = ttk.Button(self.controls_frame, text="Undo", command=self.undo_last_rectangle)
        self.undo_button.grid(row=12, column=0, pady=5, sticky="w")

        # Dropdown for detection category during drawing mode
        self.category_label = ttk.Label(self.controls_frame, text="Detection Category:")
        self.category_label.grid(row=13, column=0, pady=5, sticky="w")
        self.category_dropdown = ttk.Combobox(
            self.controls_frame, textvariable=self.current_category,
            values=["custom_bat_call", "custom_feeding_buzz", "custom_noise"], state="normal")
        self.category_dropdown.grid(row=14, column=0, pady=5, sticky="w")
        self.category_dropdown.bind("<<ComboboxSelected>>", self.validate_category)

        # Save annotations button
        self.save_button = ttk.Button(self.controls_frame, text="Save Annotations", command=self.save_annotations)
        self.save_button.grid(row=15, column=0, pady=5, sticky="w")

        # Coordinates label
        self.coordinates_label = ttk.Label(self.controls_frame, text="(x: , y: )")
        self.coordinates_label.grid(row=16, column=0, pady=5, sticky="w")

        # Detection index navigation
        self.index_label = ttk.Label(self.controls_frame, text="Go to Detection Index:")
        self.index_label.grid(row=17, column=0, pady=5, sticky="w")
        self.index_entry = ttk.Entry(self.controls_frame, width=10)
        self.index_entry.grid(row=18, column=0, pady=5, sticky="w")
        self.index_button = ttk.Button(self.controls_frame, text="Go", command=self.go_to_index)
        self.index_button.grid(row=19, column=0, pady=5, sticky="w")

        # Initialize the first plot
        self.update_plot()


    def update_index_entry(self):
        """Update the index entry field to reflect the current index."""
        self.index_entry.delete(0, tk.END)
        self.index_entry.insert(0, str(self.index+1))

    def go_to_index(self):
        """Navigate to the specified detection index."""
        try:
            entered_index = int(self.index_entry.get())-1
            if 0 <= entered_index < len(self.dataframe):
                self.index = entered_index
                self.update_plot()  # Refresh the plot for the new index
                self.update_index_entry()  # Sync the entry box with the new index
                print(f"Moved to detection index: {self.index}")
            else:
                print(f"Invalid index: {entered_index}. Must be between 0 and {len(self.dataframe) - 1}.")
        except ValueError:
            print("Please enter a valid integer for the detection index.")

    def validate_category(self):
        """Validate the entered category."""
        user_input = self.category_dropdown.get()
        predefined_categories = ["custom_bat_call", "custom_feeding_buzz", "custom_noise"]

        if user_input not in predefined_categories:
            print(f"Warning: '{user_input}' is not a predefined category. Adding as a custom category.")
        else:
            print(f"Category set to: {user_input}")

    def process_audio(self):
        """
        Process the current audio segment by applying time-stretch and resampling.
        """
        if self.audio_segment is None:
            print("No audio segment loaded.")
            return

        # Resample the audio segment to match the target sample rate
        target_sample_rate = 44100  # Match MacBook Air Speakers sample rate
        pitch_shift_semitones = 12*(self.pitch_shift.get())
        # Step 1: Apply pitch shifting without altering the duration
        pitch_shifted_audio = pitch_shift(self.audio_segment, self.fs, pitch_shift_semitones, self.nfft.get())
        # Step 2: Resample to the target sample rate for playback
        resampled_audio = librosa.resample(pitch_shifted_audio, orig_sr=self.fs, target_sr=target_sample_rate)
        # Scale to int16 if using paInt16
        self.processed_audio = (resampled_audio * 32767).astype(np.int16)
        print("Audio processed and stored for playback and saving.")

    def play_audio(self):
        """
        Play the stored processed audio segment.
        """
        self.process_audio()

        try:
            wave_obj = sa.WaveObject(self.processed_audio.tobytes(), num_channels=1, bytes_per_sample=2, sample_rate=44100)
            # Play the wave
            play_obj = wave_obj.play()
            play_obj.wait_done()  # Wait until playback is finished
            print("Playback finished.")

        except Exception as e:
            print(f"Error during playback: {e}")

    def save_audio_to_wav(self):
        """
        Save the processed audio segment to a .WAV file.
        """
        if self.processed_audio is None:
            print("No processed audio available to save.")
            return

        # Prompt user to choose a location to save the file
        file_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")],
            title="Save Audio As")

        if not file_path:
            print("Save operation canceled.")
            return

        try:
            # Write the processed audio to the chosen file
            scipy.io.wavfile.write(file_path, 44100, self.processed_audio)
            print(f"Audio saved successfully to {file_path}")
        except Exception as e:
            print(f"Error saving audio: {e}")

    def undo_last_rectangle(self):
        """Undo the last rectangle drawn."""
        if not self.drawing_stack:
            print("No rectangles to undo.")
            return
        
        # Pop the last rectangle and its associated index from the stack
        last_rectangle, last_index = self.drawing_stack.pop()
        # Remove the rectangle from the plot
        last_rectangle.remove()
        # Remove the corresponding entry from the custom_detections DataFrame
        self.custom_detections = self.custom_detections.drop(index=last_index).reset_index(drop=True)
        self.canvas.draw()
        print("Undid the last rectangle and removed its detection.")

    def set_annotation(self):
        """Update the annotation for the current spectrogram."""
        self.annotations[self.index] = self.radio_var.get()
        print(f"Set annotation for spectrogram {self.index} to {self.radio_var.get()}")

    def toggle_mode(self):
        """Toggle between Drawing and Browsing modes."""
        self.drawing_mode.set(not self.drawing_mode.get())
        if self.is_drawing_mode.get():
            self.enable_drawing()
            print("Switched to Drawing Mode.")
        else:
            self.disable_drawing()
            print("Switched to Browsing Mode.")

    def enable_drawing(self):
        """Enable interactive drawing on the spectrogram."""
        # Clear any lingering selectors
        self.clear_all_rectangle_selectors()

        # Create a new RectangleSelector
        self.rectangle_selector = RectangleSelector(
            self.ax, self.on_select, interactive=True, useblit=True,
            button=[1], minspanx=0, minspany=0, spancoords="pixels",
            props=dict(facecolor="none", edgecolor="yellow", alpha=0.8, linewidth=0.5),
            handle_props=dict(marker="o", markersize=0.5, markeredgecolor="none", markerfacecolor="none"))

        # Connect mouse motion event to update_coordinates
        self.mouse_motion_cid = self.fig.canvas.mpl_connect('motion_notify_event', self.update_coordinates)
        print("Drawing mode enabled with a fresh RectangleSelector.")

    def update_coordinates(self, event):
        """Update the coordinates label with the mouse's current position."""
        if event.inaxes == self.ax:  # Check if the mouse is within the spectrogram plot
            x, y = event.xdata, event.ydata
            row = self.dataframe.iloc[self.index]
            call_dur = (row['end_time'] - row['start_time'])
            pad = min(min(row['start_time'] - call_dur, 1795 - row['end_time']), self.duration.get())
            start = row['start_time'] - call_dur - (0.5 * pad)

            time = x
            freq = int(y)
            self.coordinates_label.config(text=f"({time:.2f}s, {freq}kHz)")
        else:
            self.coordinates_label.config(text="(x: , y: )")
            
    def disable_drawing(self):
        """Disable interactive drawing on the spectrogram."""
        # Disconnect mouse motion event
        if hasattr(self, 'mouse_motion_cid'):
            self.fig.canvas.mpl_disconnect(self.mouse_motion_cid)
        # Clear the RectangleSelector
        self.clear_all_rectangle_selectors()
        self.drawing_mode.set(False)
        print("Drawing mode disabled.")

    def clear_all_rectangle_selectors(self):
        """Clear all rectangle selectors and associated handles."""
        if hasattr(self, 'rectangle_selector') and self.rectangle_selector is not None:
            # Remove the rectangle itself
            if hasattr(self.rectangle_selector, "artists"):
                for artist in self.rectangle_selector.artists:
                    if artist in self.ax.patches:
                        artist.remove()

            # Remove the handles associated with the rectangle
            if hasattr(self.rectangle_selector, "_handles"):
                for handle in self.rectangle_selector._handles.artists:
                    if handle is not None and handle.axes:  # Check if handle is valid
                        handle.axes.remove_artist(handle)  # Remove the handle

            # Deactivate and reset the selector
            self.rectangle_selector.set_active(False)
            self.rectangle_selector = None
            self.canvas.draw()
            print("Cleared all RectangleSelector elements.")

    def on_select(self, eclick, erelease):
        """Handle rectangle drawing."""
        if not self.drawing_mode.get():
            return

        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        row = self.dataframe.iloc[self.index]
        call_dur = (row['end_time'] - row['start_time'])
        pad = min(min(row['start_time'] - call_dur, 1795 - row['end_time']), self.duration.get())
        start = row['start_time'] - call_dur - (0.5 * pad)

        start_time = start + (min(x1, x2) / (self.fs / 2))
        end_time = start + (max(x1, x2) / (self.fs / 2))
        low_freq = min(y1, y2) * (self.fs / 2)
        high_freq = max(y1, y2) * (self.fs / 2)

        # Add rectangle to custom detections with the current category
        new_detection = pd.DataFrame([{
            "input_file": row['input_file'],
            "start_time": start_time,
            "end_time": end_time,
            "low_freq": low_freq,
            "high_freq": high_freq,
            "category": self.current_category.get(),  # Assign the selected category
        }])
        self.custom_detections = pd.concat([self.custom_detections, new_detection], ignore_index=True)
        detection_index = self.custom_detections.index[-1]

        # Draw green rectangle for confirmed detection
        x = min(x1, x2)
        width = abs(x2 - x1)
        y = min(y1, y2)
        height = abs(y2 - y1)

        rect = patches.Rectangle(
            (x, y), width, height, edgecolor="green", facecolor="none", linewidth=1, alpha=0.8)
        self.ax.add_patch(rect)
        self.drawing_stack.append((rect, detection_index))  # Add rectangle and its index to the stack
        self.canvas.draw()

        print(f"Added detection: Start {start_time:.3f}s, End {end_time:.3f}s, "
              f"Low {low_freq:.1f}kHz, High {high_freq:.1f}kHz, Category: {self.current_category.get()}")

    def update_nfft(self, value):
        """Update the NFFT value and redraw the current spectrogram."""
        # Snap the slider value to the nearest power of 2
        self.nfft.set(max(256, float(value)))
        self.update_plot()

    def update_duration(self, value):
        """Update the NFFT value and redraw the current spectrogram."""
        # Snap the slider value to the nearest power of 2
        self.duration.set(float(value))
        self.update_plot()

    def update_plot(self):
        """Update the plot and checkbox state for the current spectrogram."""
        print(f"Index: {self.index}, NFFT: {self.nfft.get()}")
        row = self.dataframe.iloc[self.index]
        file_path = '/'.join(Path(row['input_file']).parts[2:])
        cleaned_path = re.sub(r"(ubna_data_\d+)_mir", r"\1", file_path)
        osn_file_path = Path(f'bio230143-bucket01/{cleaned_path}')
        if self.cur_path != osn_file_path:
            self.cur_path = osn_file_path
            print(f"Loading new file {osn_file_path}")
            file = filesys.open(path=self.cur_path)
            self.audio_file = sf.SoundFile(file)
            self.fs = self.audio_file.samplerate
        call_dur = (row['end_time'] - row['start_time'])
        pad = min(min(row['start_time'] - call_dur, 1795 - row['end_time']), self.duration.get())
        start = row['start_time'] - call_dur - (0.5 * pad)
        duration = (2 * call_dur) + (1 * pad)
        self.audio_file.seek(int(start * self.fs))
        self.audio_segment = self.audio_file.read(int(duration * self.fs))  # Numpy array
        self.audio_file.seek(int(row['start_time'] * self.fs))
        
        # Plot the spectrogram
        plot_spectrogram(self.ax,row,self.audio_segment,location_df_orig=self.location_df_kmeans_raw,test_df=self.dataframe,
            fs=self.fs,duration=duration,start=start,osn_file_path=osn_file_path,row_index=self.index,nfft=self.nfft.get())

        # Draw custom detections
        self.redraw_custom_detections(start, duration)

        # Adjust layout
        self.fig.tight_layout()

        # Redraw the canvas
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)  # Place in the plot frame
        self.canvas.get_tk_widget().pack(fill="both", expand=True)  # Use pack to fill the plot frame
        self.canvas.draw()

        # Update radiobutton state
        self.radio_var.set(self.annotations[self.index])  # Ensure this updates to the annotation for the current detection
        print(f"Updated plot for index: {self.index}, Current Detection Annotation: {self.radio_var.get()}")

        # Update the index_entry field with the current index
        self.update_index_entry()

    def redraw_custom_detections(self, start, duration):
        """Redraw custom detections on the current spectrogram."""
        visible_detections = self.custom_detections[
            (self.custom_detections["start_time"] < (start + duration)) &
            (self.custom_detections["end_time"] > start)
        ]

        for _, detection in visible_detections.iterrows():
            # Calculate rectangle coordinates relative to the current spectrogram
            rect_start = max(detection["start_time"], start)
            rect_end = min(detection["end_time"], start + duration)
            rect_low = detection["low_freq"]
            rect_high = detection["high_freq"]

            # Convert to plot coordinates
            x = (rect_start - start) * (self.fs / 2)
            width = (rect_end - rect_start) * (self.fs / 2)
            y = rect_low / (self.fs / 2)
            height = (rect_high - rect_low) / (self.fs / 2)

            # Draw the rectangle
            self.ax.add_patch(patches.Rectangle(
                (x, y), width, height, edgecolor="green", facecolor="none", linewidth=1, alpha=0.8
            ))

        print(f"Redrew {len(visible_detections)} custom detections.")

    def clear_rectangle_selector(self):
        """Clear the RectangleSelector."""
        if hasattr(self, 'rectangle_selector') and self.rectangle_selector is not None:
            self.rectangle_selector.disconnect_events()  # Disconnect events
            self.rectangle_selector.set_active(False)    # Deactivate
            self.rectangle_selector = None               # Remove reference
            print("Cleared RectangleSelector.")

    def previous_row(self):
        """Navigate to the previous spectrogram."""
        if self.index > 0:
            self.clear_rectangle_selector()  # Clear the current rectangle
            self.index -= 1
            print(f"Moved to previous row: {self.index}")
            self.update_plot()

    def next_row(self):
        """Navigate to the next spectrogram."""
        if self.index < len(self.dataframe) - 1:
            self.clear_rectangle_selector()  # Clear the current rectangle
            self.index += 1
            print(f"Moved to next row: {self.index}")
            self.update_plot()

    def save_annotations(self):
        """Save the annotated rows and custom detections."""
        bat_calls_indices = [idx for idx, annotation in self.annotations.items() if annotation == "bat_call"]
        feeding_buzzes_indices = [idx for idx, annotation in self.annotations.items() if annotation == "feeding_buzz"]
        neither_indices = [idx for idx, annotation in self.annotations.items() if annotation == "neither"]

        # Create DataFrames for each category
        bat_calls_df = self.dataframe.loc[bat_calls_indices]
        feeding_buzzes_df = self.dataframe.loc[feeding_buzzes_indices]
        neither_df = self.dataframe.loc[neither_indices]

        # Save to CSV
        bat_calls_df.to_csv(f"{Path(__file__).parent}/{Path(__file__).stem}_bat_calls.csv", index=False)
        feeding_buzzes_df.to_csv(f"{Path(__file__).parent}/{Path(__file__).stem}_feeding_buzzes.csv", index=False)
        neither_df.to_csv(f"{Path(__file__).parent}/{Path(__file__).stem}_neithers.csv", index=False)
        self.custom_detections.to_csv(f"{Path(__file__).parent}/{Path(__file__).stem}_custom_detections.csv", index=False)

        # Print confirmation
        print(f"Saved {len(bat_calls_indices)} rows to '{Path(__file__).stem}_bat_calls.csv'")
        print(f"Saved {len(feeding_buzzes_indices)} rows to '{Path(__file__).stem}_feeding_buzzes.csv'")
        print(f"Saved {len(neither_indices)} rows to '{Path(__file__).stem}_neithers.csv'")
        print(f"Saved {len(self.custom_detections)} custom detections to 'custom_detections.csv'")


def assemble_initial_location_summary(file_paths, det_threshold):
    """
    Puts together all bd2 outputs in data/raw and converts detection start_times to datetime objects.
    Returns and saves a summary of bd2-detected bat calls within a desired frequency band.
    """

    location_df = dd.read_csv(f'{file_paths["raw_SITE_folder"]}/{file_paths["detector"]}__*.csv').compute()
    if 'det_prob' in location_df.columns:
        location_df = location_df[location_df['det_prob']>=det_threshold].copy()
    location_df['start_time'] = location_df['start_time'].astype('float64')
    location_df['end_time'] = location_df['end_time'].astype('float64')
    location_df['low_freq'] = location_df['low_freq'].astype('float64')
    location_df['high_freq'] = location_df['high_freq'].astype('float64')
    file_dts = pd.to_datetime(location_df['input_file'], format='%Y%m%d_%H%M%S', exact=False)
    anchor_start_times = file_dts + pd.to_timedelta(location_df['start_time'], unit='S')
    anchor_end_times = file_dts + pd.to_timedelta(location_df['end_time'], unit='S')

    location_df.insert(0, 'call_end_time', anchor_end_times)
    location_df.insert(0, 'call_start_time', anchor_start_times)
    location_df.insert(0, 'ref_time', anchor_start_times)

    return location_df

def get_file_paths(data_params):
    """
    Assemble a dictionary for file_paths important for the pipeline.
    """

    file_paths = dict()
    file_paths["raw_SITE_folder"] = f'{Path.cwd()}/output_dir/recover-20260318/{data_params["site_name"]}'
    file_paths["detector"] = data_params['detector_tag']
    return file_paths

if __name__ == "__main__":
    data_params = dict()
    data_params["site_tag"] = 'Carp'
    data_params['type_tag'] = 'HF'
    data_params["site_name"] = SITE_NAMES[data_params["site_tag"]]
    data_params["detection_threshold"] = 0.5
    data_params["detector_tag"] = 'bd2'

    file_paths = get_file_paths(data_params)
    init_location_sum = assemble_initial_location_summary(file_paths, data_params["detection_threshold"]) 
    init_location_sum.reset_index(inplace=True)
    init_location_sum.rename({'index':'index_in_file'}, axis='columns', inplace=True)
    test_df = init_location_sum.reset_index()
    filesys = fsspec.filesystem('s3', anon=True, client_kwargs={'endpoint_url': 'https://sdsc.osn.xsede.org'})

    # Launch the GUI
    root = tk.Tk()
    root.title("Spectrogram Viewer")
    app = SpectrogramViewer(
        root,
        dataframe=test_df,
        location_df_kmeans_raw=init_location_sum)
    root.mainloop()
