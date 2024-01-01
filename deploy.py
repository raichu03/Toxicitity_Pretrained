import streamlit as st
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from wordcloud import WordCloud
import whisper
import subprocess
import base64
import os
import plotly.express as px
import matplotlib.pyplot as plt

labels = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate']


# Load the tokenizer and model from the saved directory
model_name ="results/Saved_model"
Bert_Tokenizer = BertTokenizer.from_pretrained(model_name)
Bert_Model = BertForSequenceClassification.from_pretrained(model_name)

model1 = whisper.load_model("base")

def predict_user_input(input_text, model=Bert_Model, tokenizer=Bert_Tokenizer, device=torch.device('cpu')):
    user_input = [input_text]
    user_encodings = tokenizer(user_input, truncation=True, padding=True, return_tensors="pt")
    user_dataset = TensorDataset(user_encodings['input_ids'], user_encodings['attention_mask'])
    user_loader = DataLoader(user_dataset, batch_size=1, shuffle=False)

    model.eval()
    with torch.no_grad():
        for batch in user_loader:
            input_ids, attention_mask = [t.to(device) for t in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.sigmoid(logits)

    predicted_labels = (predictions.cpu().numpy() > 0.5).astype(int)
    return predicted_labels[0].tolist()

def video2mp3(video_file, output_ext="mp3"):
    filename, ext = os.path.splitext(video_file)
    subprocess.call(["ffmpeg", "-y", "-i", video_file, f"{filename}.{output_ext}"], 
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT)
    return f"{filename}.{output_ext}"

def translate(audio):
    options = dict(beam_size=5, best_of=5)
    translate_options = dict(task="translate", **options)
    result = model1.transcribe(audio,**translate_options)
    return result

@st.cache_data
# function to display the video of a given file
def displayVideo(file):
    # Check if the file is of type UploadedFile
    if hasattr(file, 'read'):
        # If the file is an UploadedFile, read the contents
        content = file.read()
        base64_video = base64.b64encode(content).decode('utf-8')
    else:
        # If the file is a regular file path, open and read the contents
        with open(file, "rb") as f:
            base64_video = base64.b64encode(f.read()).decode('utf-8')

    # Embedding video in HTML
    display_video = F'<iframe src="data:video/mp4;base64,{base64_video}" width="100%" height="400" type="video/mp4"></iframe>'

    # Displaying Video
    st.markdown(display_video, unsafe_allow_html=True)


def main():
    st.title("BERT Sentiment Analysis and Whisper ASR")

    input_type = st.selectbox("Choose input type:", ["Text", "Video"])

    if input_type == "Text":
        user_text = st.text_area("Enter a text:", "Beautiful Smile")
        if st.button("Predict"):
            sentiment = predict_user_input(input_text=user_text)
            st.write("Sentiment:", "Negative" if sentiment[0] == 1 else "Positive")

            predicted_labels = [labels[i] for i, score in enumerate(sentiment) if score == 1]
            st.write("Predicted Labels:", ', '.join(predicted_labels))

            fig = px.bar(x=sentiment, y=labels, orientation='h', color=sentiment, color_continuous_scale='Blues')
            fig.update_layout(xaxis=dict(title='Prediction Score', range=[0, 1]), yaxis=dict(title='Labels'))
            st.plotly_chart(fig)

    elif input_type == "Video":
        video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mkv","mov"])
        
        if video_file is not None:
            if st.button("Extract Audio and Predict"):
                col1, col2 = st.columns(2)
                with col1:  
                    display_video = displayVideo(video_file)
                with col2:
                    filepath = "file/"+ video_file.name
                    with open(filepath, "wb") as temp_file:
                        temp_file.write(video_file.read())

                    audio_file = video2mp3(filepath)

                    transcription = translate(audio_file)
                    transcription_text = transcription.get("text", "")
                    st.write("Transcription:", transcription_text)

                    sentiment = predict_user_input(input_text=transcription_text)
                    st.write("Sentiment:", "Negative" if sentiment[0] == 1 else "Positive")

                    predicted_labels = [labels[i] for i, score in enumerate(sentiment) if score == 1]
                    st.write("Predicted Labels:", ', '.join(predicted_labels))

                    fig = px.bar(x=sentiment, y=labels, orientation='h', color=sentiment, color_continuous_scale='Blues')
                    fig.update_layout(xaxis=dict(title='Prediction Score', range=[0, 1]), yaxis=dict(title='Labels'))
                    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
