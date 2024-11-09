import streamlit as st
from transformers import pipeline
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from gtts import gTTS
import whisper
from pydub import AudioSegment
from io import BytesIO
from groq import Groq


class InterviewAssistant:
    def __init__(self, questions_path, groq_api_key):
        """Initialize the Interview Assistant with necessary models and data."""
        self.load_questions(questions_path)
        self.setup_models(groq_api_key)
        self.setup_faiss_index()

    def load_questions(self, questions_path):
        """Load and preprocess the questions dataset."""
        self.questions = pd.read_csv(questions_path, encoding='unicode_escape')
        self.questions = self.questions[['Question']]

    def setup_models(self, groq_api_key):
        """Initialize all required models."""
        # Sentence transformer for embeddings
        self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Text generation model
        self.generation_model = pipeline(
            "text2text-generation",
            model="google/flan-t5-large",
            num_beams=1
        )

        # Whisper model for speech-to-text
        self.whisper_model = whisper.load_model("base")

        # Groq client for LLM feedback
        self.groq_client = Groq(api_key=groq_api_key)

    def setup_faiss_index(self):
        """Create FAISS index for question similarity search."""
        embeddings = [self.sentence_model.encode(q) for q in self.questions['Question']]
        self.questions['embedding'] = embeddings
        question_embeddings = np.vstack(self.questions['embedding'].values)
        dimension = question_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(question_embeddings)

    def get_similar_questions(self, job_title, job_description, top_k=1):
        """Get similar questions based on job title and description."""
        text = f'{job_title}: {job_description}'
        text_embedding = self.sentence_model.encode(text).reshape(1, -1)
        _, indices = self.index.search(text_embedding, top_k)
        similar_questions = self.questions.iloc[indices[0]]
        return similar_questions[['Question']].to_dict(orient='records')

    def generate_feedback(self, question, user_answer):
        """Generate feedback on user's interview answer using Groq LLM."""
        response = self.groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an experienced interview coach specializing in preparing candidates "
                        "for real-world job interviews. Your goal is to provide concise, actionable "
                        "feedback that helps the user improve their answers quickly."
                    )
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\nAnswer: {user_answer}\n\n"
                              "Provide feedback on the quality of the answer, noting strengths "
                              "and suggestions for improvement."
                }
            ],
            model="llama3-8b-8192",
        )
        return response.choices[0].message.content

    def transcribe_audio(self, audio_data):
        """Convert audio data into text using Whisper."""
        audio = AudioSegment.from_wav(audio_data)
        audio.export("temp_audio.wav", format="wav")
        result = self.whisper_model.transcribe("temp_audio.wav")
        return result["text"]

    def text_to_audio(self, text):
        """Convert text to audio using gTTS."""
        tts = gTTS(text=text, lang='en')
        # Save the audio in a BytesIO object to send it directly in Streamlit
        audio_bytes = BytesIO()
        tts.save(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes


# Instantiate the assistant
assistant = InterviewAssistant(
    questions_path='non_technical_interview_questions (1).csv',
    groq_api_key="GROQ_API_KEY"
)


# Streamlit Interface
st.title("AI Interview Assistant")

# Inputs for job title and description
job_title = st.text_input("Job Title")
job_description = st.text_area("Job Description")

if st.button("Get Question"):
    # Get the similar question based on the job description
    questions = assistant.get_similar_questions(job_title, job_description)
    question_text = questions[0]['Question']

    # Display the question text
    st.text_area("Interview Question (Text)", value=question_text, height=150)

    # Convert the question to audio
    question_audio = assistant.text_to_audio(question_text)

    # Display the question in audio format
    st.audio(question_audio, format='audio/wav')


# Input for user's answer (audio format)
st.subheader("Your Answer")
audio_input = st.audio("Upload Your Audio Answer", type="file")

if audio_input:
    user_answer = assistant.transcribe_audio(audio_input)
    st.text_area("Your Answer (Text)", value=user_answer, height=150)

    if st.button("Get Feedback"):
        feedback = assistant.generate_feedback(question_text, user_answer)
        st.text_area("Feedback", value=feedback, height=150)
