

def predict_job_fit(resume_text, job_description_text):
    """Predict job fit using XGBoost"""
    # Load the trained model and tokenizer
    model_path = "output/models/xgboost-resume-job-fit/checkpoint-19"
    tokenizer = XGBoostTokenizer.from_pretrained(model_path)
    model = XGBoostForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Tokenize input text
    inputs = tokenizer(resume_text, job_description_text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs)
    
    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    
    # Get the index of the most probable label
    predicted_label = torch.argmax(probabilities, dim=1).item()
    
    # Map label index to string
    label_map = {0: "No Fit", 1: "Fit"}
    predicted_label_str = label_map[predicted_label]
    
    return predicted_label_str