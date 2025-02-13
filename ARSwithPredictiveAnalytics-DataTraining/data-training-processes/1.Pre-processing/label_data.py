import pandas as pd
import json
import os
from typing import List, Dict
import nltk
from nltk.tokenize import word_tokenize
import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
from datetime import datetime

# Download required NLTK data
nltk.download('punkt')

class EntityLabeler:
    def __init__(self, master):
        self.master = master
        self.master.title("Resume Entity Labeler")
        self.master.geometry("1200x800")

        # Entity types
        self.entity_types = [
            "AGE", "GENDER", "ADDRESS", "SKILL",
            "EDUCATION", "EXPERIENCE", "CERTIFICATION", "O"
        ]
        
        self.current_resume_idx = 0
        self.labeled_data = []
        self.tokens = []
        self.labels = []
        
        self.setup_ui()
        self.load_data()

    def setup_ui(self):
        # Top frame for text display
        top_frame = ttk.Frame(self.master)
        top_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Text display
        self.text_display = ScrolledText(top_frame, wrap=tk.WORD, width=80, height=20)
        self.text_display.pack(fill=tk.BOTH, expand=True)

        # Middle frame for token labeling
        middle_frame = ttk.Frame(self.master)
        middle_frame.pack(fill=tk.X, padx=10, pady=5)

        # Token display
        self.token_label = ttk.Label(middle_frame, text="Current token: ")
        self.token_label.pack(side=tk.LEFT)

        # Entity type selection
        self.entity_var = tk.StringVar()
        self.entity_dropdown = ttk.Combobox(middle_frame, textvariable=self.entity_var, values=self.entity_types)
        self.entity_dropdown.pack(side=tk.LEFT, padx=5)
        self.entity_dropdown.set("O")

        # Bottom frame for buttons
        bottom_frame = ttk.Frame(self.master)
        bottom_frame.pack(fill=tk.X, padx=10, pady=5)

        # Buttons
        ttk.Button(bottom_frame, text="Label Token", command=self.label_token).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_frame, text="Previous Token", command=self.prev_token).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_frame, text="Next Token", command=self.next_token).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_frame, text="Save Progress", command=self.save_progress).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_frame, text="Next Resume", command=self.next_resume).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_frame, text="Skip Resume", command=self.skip_resume).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_frame, text="Show Stats", command=self.show_progress_stats).pack(side=tk.LEFT, padx=5)

        # Progress display
        self.progress_var = tk.StringVar()
        ttk.Label(bottom_frame, textvariable=self.progress_var).pack(side=tk.RIGHT, padx=5)

    def load_data(self):
        try:
            # Load preprocessed resumes
            csv_path = "dataset/3.pre-processed/csv-files/resumes.csv"
            self.df = pd.read_csv(csv_path)
            
            # Load existing progress if any
            json_path = "output/dataset/labeled_data.json"
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    self.labeled_data = json.load(f)
                    
                # Find the last annotated resume index
                if self.labeled_data:
                    last_annotated = max(item['resume_idx'] for item in self.labeled_data)
                    self.current_resume_idx = last_annotated + 1
                else:
                    self.current_resume_idx = 0
            else:
                self.labeled_data = []
                self.current_resume_idx = 0
            
            self.load_current_resume()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading data: {str(e)}")

    def load_current_resume(self):
        if self.current_resume_idx < len(self.df):
            text = self.df.iloc[self.current_resume_idx]['preprocessed_text']
            self.text_display.delete('1.0', tk.END)
            self.text_display.insert('1.0', text)
            
            # Tokenize text
            self.tokens = word_tokenize(text)
            self.labels = ['O'] * len(self.tokens)
            self.current_token_idx = 0
            self.update_token_display()
            self.update_progress()

    def update_token_display(self):
        if self.current_token_idx < len(self.tokens):
            token = self.tokens[self.current_token_idx]
            label = self.labels[self.current_token_idx]
            self.token_label.config(text=f"Current token: {token} ({label})")
            
            # Highlight current token in text
            self.highlight_current_token()

    def highlight_current_token(self):
        # Remove previous highlighting
        self.text_display.tag_remove('highlight', '1.0', tk.END)
        
        # Find and highlight current token
        text = self.text_display.get('1.0', tk.END)
        token = self.tokens[self.current_token_idx]
        start = '1.0'
        
        # Count tokens until current token
        count = 0
        for i in range(self.current_token_idx):
            count += len(self.tokens[i]) + 1
        
        # Calculate positions
        start = f"1.{count}"
        end = f"1.{count + len(token)}"
        
        # Add highlighting
        self.text_display.tag_add('highlight', start, end)
        self.text_display.tag_config('highlight', background='yellow')

    def label_token(self):
        if self.current_token_idx < len(self.tokens):
            entity_type = self.entity_var.get()
            self.labels[self.current_token_idx] = f"B-{entity_type}" if entity_type != "O" else "O"
            self.next_token()

    def next_token(self):
        if self.current_token_idx < len(self.tokens) - 1:
            self.current_token_idx += 1
            self.update_token_display()
            self.update_progress()

    def prev_token(self):
        if self.current_token_idx > 0:
            self.current_token_idx -= 1
            self.update_token_display()
            self.update_progress()

    def save_progress(self):
        labeled_example = {
            'tokens': self.tokens,
            'labels': self.labels,
            'resume_idx': self.current_resume_idx
        }
        self.labeled_data.append(labeled_example)
        
        # Save to JSON file
        json_path = "output/dataset/labeled_data.json"
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(self.labeled_data, f, indent=2)
        
        messagebox.showinfo("Success", "Progress saved successfully!")

    def next_resume(self):
        self.save_progress()
        self.current_resume_idx += 1
        if self.current_resume_idx < len(self.df):
            self.load_current_resume()
        else:
            messagebox.showinfo("Completed", "All resumes have been labeled!")
            self.master.quit()

    def skip_resume(self):
        """
        Skip the current resume without saving and move to the next one
        """
        skip_response = messagebox.askyesno(
            "Skip Resume",
            "Are you sure you want to skip this resume without saving?"
        )
        
        if skip_response:
            # Record skipped resume
            skipped_info = {
                'resume_idx': self.current_resume_idx,
                'filename': self.df.iloc[self.current_resume_idx]['filename'],
                'status': 'skipped',
                'skip_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Create or update skipped resumes log
            skip_log_path = "output/dataset/skipped_resumes.json"
            os.makedirs(os.path.dirname(skip_log_path), exist_ok=True)
            
            skipped_resumes = []
            if os.path.exists(skip_log_path):
                with open(skip_log_path, 'r') as f:
                    skipped_resumes = json.load(f)
            
            skipped_resumes.append(skipped_info)
            
            with open(skip_log_path, 'w') as f:
                json.dump(skipped_resumes, f, indent=2)
            
            # Move to next resume
            self.current_resume_idx += 1
            if self.current_resume_idx < len(self.df):
                self.load_current_resume()
            else:
                messagebox.showinfo("Completed", "All resumes have been processed!")
                self.master.quit()

    def show_progress_stats(self):
        """
        Show statistics about labeled and skipped resumes
        """
        total_resumes = len(self.df)
        labeled_count = len(self.labeled_data)
        
        # Count skipped resumes
        skip_log_path = "output/dataset/skipped_resumes.json"
        skipped_count = 0
        if os.path.exists(skip_log_path):
            with open(skip_log_path, 'r') as f:
                skipped_resumes = json.load(f)
                skipped_count = len(skipped_resumes)
        
        remaining = total_resumes - labeled_count - skipped_count
        
        stats_message = f"""
        Progress Statistics:
        - Total Resumes: {total_resumes}
        - Labeled: {labeled_count}
        - Skipped: {skipped_count}
        - Remaining: {remaining}
        - Current Resume: {self.current_resume_idx + 1}
        """
        
        messagebox.showinfo("Progress Statistics", stats_message)

    def update_progress(self):
        total = len(self.tokens)
        current = self.current_token_idx + 1
        resume_progress = f"Resume {self.current_resume_idx + 1}/{len(self.df)}"
        token_progress = f"Token {current}/{total}"
        self.progress_var.set(f"{resume_progress} - {token_progress}")

if __name__ == "__main__":
    root = tk.Tk()
    app = EntityLabeler(root)
    root.mainloop()
