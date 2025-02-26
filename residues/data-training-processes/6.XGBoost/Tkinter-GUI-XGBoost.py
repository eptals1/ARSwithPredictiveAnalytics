import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
# import json
import xgboost as xgb
import re

class ResumeEvaluatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Resume Evaluator")

        tk.Label(root, text="Job Role:").grid(row=0, column=0)
        self.job_role_entry = tk.Entry(root)
        self.job_role_entry.grid(row=0, column=1)

        # Job Requirement Inputs
        tk.Label(root, text="Age Min:").grid(row=1, column=0)
        self.age_min_entry = tk.Entry(root)
        self.age_min_entry.grid(row=1, column=1)

        tk.Label(root, text="Age Max:").grid(row=2, column=0)
        self.age_max_entry = tk.Entry(root)
        self.age_max_entry.grid(row=2, column=1)

        tk.Label(root, text="Preferred Gender:").grid(row=3, column=0)
        self.gender_entry = tk.Entry(root)
        self.gender_entry.grid(row=3, column=1)

        tk.Label(root, text="Location:").grid(row=4, column=0)
        self.location_entry = tk.Entry(root)
        self.location_entry.grid(row=4, column=1)

        tk.Label(root, text="Skills (comma-separated):").grid(row=5, column=0)
        self.skills_entry = tk.Entry(root)
        self.skills_entry.grid(row=5, column=1)

        tk.Label(root, text="Education:").grid(row=6, column=0)
        self.education_entry = tk.Entry(root)
        self.education_entry.grid(row=6, column=1)

        tk.Label(root, text="Experience (years):").grid(row=7, column=0)
        self.experience_entry = tk.Entry(root)
        self.experience_entry.grid(row=7, column=1)

        tk.Label(root, text="Certifications (comma-separated):").grid(row=8, column=0)
        self.certifications_entry = tk.Entry(root)
        self.certifications_entry.grid(row=8, column=1)

        # Buttons
        self.load_resumes_button = tk.Button(root, text="Load Resumes", command=self.load_resumes)
        self.load_resumes_button.grid(row=8, column=0, columnspan=2, pady=5)

        self.evaluate_button = tk.Button(root, text="Evaluate", command=self.evaluate_resumes)
        self.evaluate_button.grid(row=9, column=0, columnspan=2, pady=5)

        # Result Textbox
        self.result_text = tk.Text(root, height=10, width=60)
        self.result_text.grid(row=10, column=0, columnspan=2, pady=5)

        self.resumes = []

    def load_resumes(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx;*.xls")])
        if file_path:
            df = pd.read_excel(file_path)

            # Convert DataFrame to list of dictionaries (resumes)
            self.resumes = df.to_dict(orient="records")

            messagebox.showinfo("Success", "Resumes loaded successfully!")

    def extract_experience_in_months(experience_str):
        """Converts experience text (e.g., '2 years 3 months') into months."""
        years = months = 0
        match = re.search(r"(\d+)\s*years?", experience_str)
        if match:
            years = int(match.group(1)) * 12  # Convert years to months

        match = re.search(r"(\d+)\s*months?", experience_str)
        if match:
            months = int(match.group(1))  # Keep months as is

        return years + months  # Total months

    def preprocess_resume(self, resume, job):
        resume_experience_months = extract_experience_in_months(resume["Experience"])
        job_experience_months = extract_experience_in_months(job["experience_required"])

        return {
            "age": resume["Age"],
            "age_match": int(job["age_min"] <= resume["Age"] <= job["age_max"]),
            "gender_match": int(resume["Gender"] == job["preferred_gender"]),
            "address_match": int(resume["Address"] == job["location"]),
            "skill_match": int(any(skill in job["skills_required"] for skill in resume["Skills"].split(","))),
            "education_match": int(resume["Education"] == job["education_required"]),
            "experience": resume_experience_months,
            "experience_match": int(resume_experience_months >= job_experience_months),
            "certification_match": int(any(cert in job["certifications_required"] for cert in resume["Certification"].split(",")))
        }


    def evaluate_resumes(self):
        if not self.resumes:
            messagebox.showerror("Error", "No resumes loaded!")
            return

        # Get job requirements
        job_requirements = {
            "job_role": self.job_role_entry.get(),
            "age_min": int(self.age_min_entry.get()),
            "age_max": int(self.age_max_entry.get()),
            "preferred_gender": self.gender_entry.get(),
            "location": self.location_entry.get(),
            "skills_required": self.skills_entry.get().split(","),
            "education_required": self.education_entry.get(),
            "experience_required": self.experience_entry.get(),
            "certifications_required": self.certifications_entry.get().split(","),
        }

        # Convert resumes to structured data
        X = pd.DataFrame([self.preprocess_resume(resume, job_requirements) for resume in self.resumes])

        # Dummy labels for training
        y = [1 if resume["experience"] >= job_requirements["experience_required"] else 0 for resume in self.resumes]

        # Train XGBoost model
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X, y)

        # Predict suitability scores
        y_pred_proba = model.predict_proba(X)[:, 1]
        suitability_scores = (y_pred_proba * 100).astype(int)

        # Rank candidates
        candidates = []
        for i, resume in enumerate(self.resumes):
            status = "Suitable" if suitability_scores[i] >= 70 else "Not Suitable"
            candidates.append((resume["resume_id"], suitability_scores[i], status))

        candidates.sort(key=lambda x: x[1], reverse=True)

        # Display results
        self.result_text.insert(tk.END, f"Job Role: {job_requirements['job_role']}\n\n")
        for candidate in candidates:
            self.result_text.insert(tk.END, f"{candidate[0]}: {candidate[1]}% ({candidate[2]})\n")

# Run Tkinter App
if __name__ == "__main__":
    root = tk.Tk()
    app = ResumeEvaluatorApp(root)
    root.mainloop()
