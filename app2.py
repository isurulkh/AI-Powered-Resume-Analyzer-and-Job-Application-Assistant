import gradio as gr
import fitz
from PIL import Image
import io
import json
import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()


INTERMEDIATE_JSON_PATH = "intermediate_data.json"
INTERMEDIATE_JOB_DESC_PATH = "intermediate_job_desc.txt"

# Define a custom theme for the interface
custom_theme = {
    "primary_color": "#FF4B4B",
    "secondary_color": "#FFD3D3",
    "text_color": "#333333",
    "background_color": "#FFFFFF",
    "container_color": "#F8F8F8",
    "border_color": "#EAEAEA",
}

def load_prompt(filename):
    """Function to load a prompt from a file."""
    try:
        with open(filename, "r") as file:
            return file.read()
    except Exception as e:
        return f"Error loading prompt: {e}"

def process_pdf_and_save_job_desc(pdf_file, job_description, api_key):
    if not pdf_file:
        return None, "No file provided"

    # Configure the Gemini model using the provided API key
    genai.configure(api_key=api_key)
    model_vision = genai.GenerativeModel('gemini-pro-vision')
    model_text = genai.GenerativeModel("gemini-pro")

    doc = fitz.open(stream=pdf_file, filetype="pdf")

    # Store results in a list and process all pages
    json_data = []
    images = []  # List to hold images of each page
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))
        images.append(image)

        # ... Your image processing with the genai model
        prompt = load_prompt("prompts/resume_parsing_prompt.txt")
        response = model_vision.generate_content([prompt, image])
        json_data.append(response.text) 

    doc.close()

    # Store data appropriately (consider a list of JSON objects, or a structured dict)
    with open(INTERMEDIATE_JSON_PATH, "w", encoding='utf-8') as json_file:  # Specify UTF-8 encoding
        json.dump(json_data, json_file)
    with open(INTERMEDIATE_JOB_DESC_PATH, "w", encoding='utf-8') as file:  # Specify UTF-8 encoding
        file.write(job_description)

    return images, json_data  # Return the list of images

def display_json():
    try:
        with open(INTERMEDIATE_JSON_PATH, "r") as json_file:
            json_data = json.load(json_file)
        return json.dumps(json_data, indent=4)
    except FileNotFoundError:
        return "No data available yet. Please run the first tab."

def generate_content_based_on_json(example_functionality):
    """
    Placeholder function to demonstrate generating content based on JSON data.
    Replace 'example_functionality' with actual logic for generating interview questions,
    cover letters, or skill gap analysis.
    """
    try:
        with open(INTERMEDIATE_JSON_PATH, "r") as json_file:
            json_data = json.load(json_file)
        with open(INTERMEDIATE_JOB_DESC_PATH, "r") as file:
            job_description = file.read()

        # Placeholder: Generate content based on JSON data and job description
        generated_content = f"Generated content for {example_functionality}."
        return generated_content

    except Exception as e:
        return f"An error occurred: {e}"

def generate_interview_questions(api_key):
    with open(INTERMEDIATE_JSON_PATH, "r") as json_file:
        json_data = json.load(json_file)

    combined_data = " ".join(json_data)  # Combine with spaces (adjust as needed)
    prompt = load_prompt("prompts/interview_questions_prompt.txt") + combined_data

    # Configure the Gemini model using the provided API key
    genai.configure(api_key=api_key)
    model_text = genai.GenerativeModel("gemini-pro")

    responses = model_text.generate_content(prompt)
    return responses.text

# Define the new Gradio interface for generating interview questions with API key input
interview_interface = gr.Interface(
    fn=generate_interview_questions,
    inputs=[gr.Textbox(label="Gemini API Key", placeholder="Enter your Gemini API key here...")],
    outputs=gr.Textbox(label="Generated Interview Questions"),
    title="Generate Interview Questions"
)


def generate_skill_gap_analysis(api_key):
    try:
        # Read the saved resume data (JSON)
        with open(INTERMEDIATE_JSON_PATH, "r") as file:
            json_data = file.read()

        # Read the saved job description
        with open(INTERMEDIATE_JOB_DESC_PATH, "r") as file:
            job_description = file.read()

        # Configure the Gemini model using the provided API key
        genai.configure(api_key=api_key)
        model_text = genai.GenerativeModel("gemini-pro")

        # Construct a detailed prompt for the Gemini model
        prompt = load_prompt("prompts/skills_gap_prompt.txt").replace(
            "job_description", job_description).replace("json_data", json_data)
        # Call the Gemini model to generate the skill gap analysis
        response = model_text.generate_content(prompt)

        # Format and return the skill gap analysis
        return response.text

    except Exception as e:
        return f"An error occurred: {e}"

# Define the Gradio interface for generating a skill gap analysis with API key input
skill_gap_analysis_interface = gr.Interface(
    fn=generate_skill_gap_analysis,
    inputs=[gr.Textbox(label="Gemini API Key", placeholder="Enter your Gemini API key here...")],
    outputs=gr.Textbox(label="Skill Gap Analysis"),
    title="Skill Gap Analysis"
)


def generate_cover_letter(api_key):
    try:
        # Configure the Gemini model using the provided API key
        genai.configure(api_key=api_key)
        model_text = genai.GenerativeModel("gemini-pro")

        # Read the saved job description
        with open(INTERMEDIATE_JOB_DESC_PATH, "r") as file:
            job_description = file.read()

        # Read the saved resume data (JSON)
        with open(INTERMEDIATE_JSON_PATH, "r") as file:
            json_data = file.read()

        # Create a prompt for the cover letter
        prompt = load_prompt("prompts/cover_letter_prompt.txt").replace(
            "job_description", job_description).replace("json_data", json_data)

        # Generate the cover letter using the model
        response = model_text.generate_content(prompt)

        return response.text

    except Exception as e:
        return f"An error occurred: {e}"

# Define the Gradio interface for generating a cover letter with API key input
cover_letter_interface = gr.Interface(
    fn=generate_cover_letter,
    inputs=[gr.Textbox(label="Gemini API Key", placeholder="Enter your Gemini API key here...")],
    outputs=gr.Textbox(label="Generated Cover Letter"),
    title="Cover Letter Generator"
)


def gradio_pdf_interface(pdf_content, job_description, api_key):
    images, _ = process_pdf_and_save_job_desc(pdf_content, job_description, api_key)
    return images  # Return the list of images to be displayed in the Gallery

# Define the updated interface for PDF processing with an additional input for the API key
pdf_interface = gr.Interface(
    fn=gradio_pdf_interface,
    inputs=[
        gr.File(type="binary", label="Upload PDF Resume"),
        gr.Textbox(label="Job Description", placeholder="Enter the job description here..."),
        gr.Textbox(label="Gemini API Key", placeholder="Enter your Gemini API key here...")
    ],
    outputs=gr.Gallery(label="Processed PDF Pages"),
    title="PDF Processing and Job Description",
    description="Upload a PDF resume, provide the job description, and enter your Gemini API key. The system will process the resume and extract relevant data."
)


json_interface = gr.Interface(
    fn=display_json,
    inputs=[],
    outputs=gr.Textbox(label="Resume Data in JSON", lines=20),
    title="Display JSON",
    description="View the extracted resume data in JSON format."
)

interview_interface = gr.Interface(
    fn=generate_interview_questions,
    inputs=[],
    outputs=gr.Textbox(label="Generated Interview Questions"),
    title="Generate Interview Questions"
)

skill_gap_analysis_interface = gr.Interface(
    fn=generate_skill_gap_analysis,
    inputs=[],
    outputs=gr.Textbox(label="Skill Gap Analysis"),
    title="Skill Gap Analysis"
)

cover_letter_interface = gr.Interface(
    fn=generate_cover_letter,
    inputs=[],
    outputs=gr.Textbox(label="Generated Cover Letter"),
    title="Cover Letter Generator"
)

# Combine interfaces into a TabbedInterface with improved UI/UX
demo = gr.TabbedInterface(
    [pdf_interface, json_interface, interview_interface, skill_gap_analysis_interface, cover_letter_interface],
    ["Process PDF", "JSON Output", "Interview Questions", "Skill Gap Analysis", "Cover Letter"],
    css="""
        body { font-family: Arial, sans-serif; }
        .tab { font-weight: bold; background-color: #FFD3D3; color: #333333; border-color: #EAEAEA; }
        .tab.selected { background-color: #FF4B4B; }
        .input_interface { margin-bottom: 15px; }
        .output_interface { margin-top: 15px; }
    """
)



if __name__ == "__main__":
    demo.launch()
