import os
import gradio as gr
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

question_types = {
    "MCQs": "Generate multiple-choice questions from the given content.",
    "Short Answer": "Generate short-answer questions from the given content.",
    "Interview": "Generate interview-style questions from the given content."
}

def question_generator(content, q_type):
    system_prompt = question_types[q_type]
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.4,
            max_output_tokens=2000
        ),
        contents=content
    )
    return response.text

demo = gr.Interface(
    fn=question_generator,
    inputs=[
        gr.Textbox(
            lines=6,
            placeholder="Paste study material or content here...",
            label="Input Content"
        ),
        gr.Radio(
            choices=list(question_types.keys()),
            value="MCQs",
            label="Question Type"
        )
    ],
    outputs=gr.Textbox(lines=12, label="Generated Questions"),
    title="Question Generator",
    description="Generate MCQs, short-answer, or interview-style questions from given content using Gemini."
)

demo.launch(debug=True)
