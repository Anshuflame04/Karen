from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware

# Configure API key
api_key = "AIzaSyCRObA-BWBi7GBNNq6DBuWeJe7HM3o0Vrw"
genai.configure(api_key=api_key)

# Create FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your React Native app URL for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Gemini model
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

@app.post("/analyze-image/")
async def analyze_image(file: UploadFile = File(...)):
    # Read the uploaded image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # Set the prompt for the Gemini model
    prompt = "You are a blind assistant. Analyze the objects in the scene and describe its content precisely.Like the objects detected and their relationship."

    try:
        # Call the Gemini model with the prompt and the image
        response = model.generate_content([prompt, image])

        # Return the model's response
        return JSONResponse(content={"description": response.text.strip()})
    except Exception as e:
        # Handle any errors that occur during processing
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Anshu Gemini API"}
