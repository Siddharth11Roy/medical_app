import os
import re
import tempfile
import base64
from flask import Flask, render_template, request, redirect, url_for
from google import genai
from google.genai import types
from pdf2image import convert_from_path
from PIL import Image
from dotenv import load_dotenv

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()
# The Gemini API Key should be stored as 'GEMINI_API_KEY' in .env and Render
API_KEY = os.getenv("GEMINI_API_KEY") 

if not API_KEY:
    print("Warning: GEMINI_API_KEY not found. Please set it in your environment.")
    # Client creation will likely fail or require the env var to be set elsewhere
    os.environ["GEMINI_API_KEY"] = "x" 

# Initialize the Gemini Client
# The client automatically picks up the GEMINI_API_KEY environment variable.
try:
    client = genai.Client()
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    # Handle case where client cannot be initialized (e.g., API key missing)
    pass


# Use gemini-2.5-flash for its speed and multimodal capabilities
GEMINI_MODEL = "gemini-2.5-flash" 
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir() # Use system temp folder

# --- Helper Functions (pdf_to_images, combine_images_vertically, cleanup_temp_files, clean_result remain mostly the same) ---

def pdf_to_images(pdf_file_storage, output_folder=tempfile.gettempdir(), dpi=300):
    """
    Converts an uploaded Flask FileStorage object (PDF) into images.
    Returns a list of temporary image file paths.
    """
    if not pdf_file_storage:
        raise ValueError("No PDF file provided.")

    temp_pdf_path = os.path.join(output_folder, next(tempfile._get_candidate_names()) + ".pdf")
    pdf_file_storage.save(temp_pdf_path)

    image_paths = []
    try:
        images = convert_from_path(temp_pdf_path, dpi=dpi)
        for i, image in enumerate(images):
            image_path = os.path.join(output_folder, f"page_{i + 1}_{next(tempfile._get_candidate_names())}.png")
            image.save(image_path, "PNG")
            image_paths.append(image_path)
    finally:
        os.remove(temp_pdf_path)

    return image_paths


def combine_images_vertically(image_paths, output_folder=tempfile.gettempdir()):
    """
    Combines multiple images vertically into a single composite image and saves it to a temp path.
    Returns the path to the composite image.
    """
    if not image_paths:
        return None
        
    images = [Image.open(image) for image in image_paths]
    combined_width = max(image.width for image in images)
    combined_height = sum(image.height for image in images)

    composite_image = Image.new("RGB", (combined_width, combined_height))
    y_offset = 0
    for image in images:
        composite_image.paste(image, (0, y_offset))
        y_offset += image.height
        image.close() # Close image files after processing

    composite_path = os.path.join(output_folder, f"composite_{next(tempfile._get_candidate_names())}.png")
    composite_image.save(composite_path, "PNG")
    
    return composite_path


def cleanup_temp_files(paths):
    """Removes temporary files after use."""
    for path in paths:
        try:
            os.remove(path)
        except OSError:
            print(f"Could not remove temporary file: {path}")


def clean_result(result):
    """Cleans the model response."""
    cleaned_result = re.sub(r"[^\x00-\x7F]+", "", result)
    return cleaned_result.strip()


def encode_image_to_base64(image_path):
    """
    Encodes an image file to a Base64 string for the template display (Gemini API doesn't need this, 
    but the template still uses it to display the processed image).
    """
    with open(image_path, "rb") as image_file:
        return f"data:image/png;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"

# --- CORE GEMINI API CALL (Replaced OpenAI call) ---

def analyze_image(image_path, section_prompt):
    """General function to call Gemini API with a specific prompt and image."""
    try:
        # Create the image part from the file path
        # Using PIL to open the image and then passing it directly is the recommended way
        img = Image.open(image_path)
        
        # Construct the content list: [prompt_text, image_part]
        content_parts = [section_prompt, img]

        # Call the Gemini API
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=content_parts,
            config=types.GenerateContentConfig(
                temperature=0.7
            )
        )
        return clean_result(response.text)
        
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return f"Error: Could not analyze report. API issue: {str(e)}"

# --- Section Prompts (Remain Identical) ---
# NOTE: The prompt definitions are omitted here for brevity but should be included
# exactly as they were in the previous complete app.py file.
HEALTH_SUMMARY_PROMPT = """
You are going to facilitate the Report Rx tech, an innovative medical report analyzer which has its approach implemented by a dynamic consortium of virtual experts, each serving a distinct role.
Your role will be the Medical Expert (ME), a professional who has all knowledge with respect to the healthcare.. As the ME, you will facilitate the report rx process by following these key stages
HEALTH SUMMARY - Summarize the report findings by:
Highlighting key observations (e.g., normal and abnormal values).
Mentioning overall health implications (e.g., "Overall health is satisfactory, but some parameters indicate potential liver dysfunction and risk of anemia").
Include a general assessment of whether the report suggests any immediate health concerns.
Only mention levels of those parameters in this which seem abnormal. Also instead of this section being the final output it should be the intial summary which comforts the user/patient
Example:Overall, the report indicates satisfactory health. Blood sugar levels and kidney functions are within normal ranges. However, hemoglobin levels (11.2 g/dL, below the reference range of 12-15 g/dL) suggest mild anemia, and SGPT levels (56 IU/L, above the reference range of 7-40 IU/L) point to potential liver stress.
"""
GLANCE_PARAMETERS_PROMPT = """
Glance at Important Parameters - 
List critical health parameters, organizing them by category (e.g., Glucose, Liver Function, Lipid Profile, etc.), and flag those that are abnormal or near-boundary values.
Example:
Glucose: 
Fasting Glucose: 89 mg/dL (Normal)
HbA1c: 5.6% (Normal, but approaching pre-diabetes threshold)
Liver Function:
SGPT (ALT): 56 IU/L (High, above 7-40 IU/L)
SGOT (AST): 42 IU/L (Slightly High, above 10-40 IU/L)
Blood Count:
Hemoglobin: 11.2 g/dL (Low, reference range: 12-15 g/dL)
Lipid Profile:
Total Cholesterol: 230 mg/dL (High, above 200 mg/dL)
Under each test , this should cover all the paramters and if any parameter is abornal it should be mentioned in bold.
"""
POTENTIAL_RISKS_PROMPT = """
POTENTIAL RISKS
This should start with lines like (" since ur report has abnormalities such as mention the abnormals with range ( verbose 1)
Followed by this it should mention thr risks such as if glucose is high then maybe diabetes or if lymphocyte count is high then it may point to any infection somewhere in the body. ( Verbose-2)
After this there should be a nested sub-heading - " WHY ARE U PRONE TO THESE DISEASES "
This subheading should be detailed and should contain the following info -
For each flagged/abmnormal parameter, provide:
Explanation of the Parameter: Explain the role of the parameter in health (e.g., "Hemoglobin is essential for oxygen transport in the blood").
Observed Value and Reference Range: Clearly state the observed value and the normal range.
Health Implications: Explain what deviations mean (e.g., "Low hemoglobin may indicate anemia, possibly due to iron deficiency").
Actions not performed - mention all those actions performed by the patient  / lifestyle that might be carried by the patient / intakes that the patient is missing due to which the above parameter is abnoraml
Example:
Hemoglobin (11.2 g/dL, Low):
Explanation: Hemoglobin is a protein in red blood cells that carries oxygen to the body.
Implications: Low levels suggest anemia, which could result in fatigue, weakness, or more serious complications if untreated.
Unintended actions - iron deficient diet , less exercise etc
"""
DIET_RECOMMENDATIONS_PROMPT = """
Diet Do's and Don'ts:
Provide dietary recommendations based on abnormal parameters. Categorize them into foods to include and avoid, targeting the specific issues flagged.
Example:
For Anemia:
Foods to Include: Spinach, lentils, tofu, red meat, dates, vitamin C-rich foods like oranges (to enhance iron absorption).
Foods to Avoid: Tea and coffee near meals (reduce iron absorption).
For High Cholesterol:
Foods to Include: Oats, flaxseeds, nuts (almonds, walnuts), fatty fish (salmon, mackerel), olive oil.
Foods to Avoid: Fried foods, processed meats, butter, and cheese.
"""
CONSOLIDATED_GUIDANCE_PROMPT = """ 
Provide a summary of recommendations for overall health improvement, considering the combined impact of the report findings. Include:
Medical follow-up advice (e.g., "Consult a gastroenterologist for liver enzyme abnormalities").
This should convince the user that although there is nothing to worry about and should consult a specialist if possible to ensure safety.
"""
FINAL_SUMMARY_PROMPT = """
FINAL SUMMARY - Just a detailed summary of everything above that seems satisfactory to the user/patient."""
# Add the actual prompt text back here!

# --- Flask Routes (Remain Identical in Structure) ---

@app.route('/', methods=['GET'])
def index():
    """Renders the main upload form page."""
    # ... (code omitted for brevity)
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """Handles PDF upload, processing, and sends results to the display page."""
    uploaded_file = request.files.get('pdf_file')
    
    if not uploaded_file or uploaded_file.filename == '':
        return redirect(url_for('index', error="No file selected."))

    temp_paths = [] 
    
    try:
        # 1. Convert PDF to images
        image_paths = pdf_to_images(uploaded_file, output_folder=app.config['UPLOAD_FOLDER'])
        temp_paths.extend(image_paths)

        # 2. Combine images
        composite_image_path = combine_images_vertically(image_paths, output_folder=app.config['UPLOAD_FOLDER'])
        temp_paths.append(composite_image_path)
        
        # 3. Analyze the composite image (calls to Gemini)
        results = {
            'health_summary': analyze_image(composite_image_path, HEALTH_SUMMARY_PROMPT),
            'glance_parameters': analyze_image(composite_image_path, GLANCE_PARAMETERS_PROMPT),
            'potential_risks': analyze_image(composite_image_path, POTENTIAL_RISKS_PROMPT),
            'diet_recommendations': analyze_image(composite_image_path, DIET_RECOMMENDATIONS_PROMPT),
            'consolidated_guidance': analyze_image(composite_image_path, CONSOLIDATED_GUIDANCE_PROMPT),
            'final_summary': analyze_image(composite_image_path, FINAL_SUMMARY_PROMPT)
        }
        
        # 4. Encode the composite image for display
        composite_image_b64 = encode_image_to_base64(composite_image_path)
        
        return render_template('index.html', results=results, composite_image_b64=composite_image_b64)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return render_template('index.html', error=f"An error occurred during processing: {str(e)}")
        
    finally:
        # 5. Cleanup all temporary files
        cleanup_temp_files(temp_paths)


if __name__ == '__main__':
    app.run(debug=True)