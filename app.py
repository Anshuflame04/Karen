from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import numpy as np
import asyncio
import cv2
import base64
import json
from ultralytics import YOLO
import google.generativeai as genai

# Configure Gemini API
api_key = "AIzaSyBSCp3SG9pBiAFvo9e5zVupU4D4Nhoyd-o"
genai.configure(api_key=api_key)

generation_config = {
    "temperature": 0.1,
    "top_p": 0.9,
    "top_k": 20,
    "max_output_tokens": 512,
    "response_mime_type": "text/plain",
}

# Load Gemini model
model_gemini = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

def generate_content(prompt):
    try:
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        print("Error with Gemini API:", str(e))
        return "I encountered an error while processing your request."

# Load YOLO model
model_yolo = YOLO('yolov10n.pt')

def process_frame(frame):
    results = model_yolo(frame)
    detected_objects = []

    for box in results[0].boxes:
        xyxy = box.xyxy.cpu().numpy()
        confidence = float(box.conf.cpu().numpy().item())
        class_id = int(box.cls.cpu().numpy().item())
        label = model_yolo.names[class_id]

        x_min, y_min, x_max, y_max = map(int, xyxy.flatten())
        object_center_x = (x_min + x_max) // 2
        object_center_y = (y_min + y_max) // 2
        horizontal_position = "left" if object_center_x < frame.shape[1] // 2 else "right"
        vertical_position = "up" if object_center_y < frame.shape[0] // 2 else "down"
        position_description = f"{label} ({confidence:.2f}) is {horizontal_position} and {vertical_position}"
        detected_objects.append(position_description)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return frame, detected_objects

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)  # Expecting JSON with "image" and "prompt"
            image_data = payload.get("image")
            user_prompt = payload.get("prompt", "")

            if not image_data:
                await websocket.send_text(json.dumps({"error": "No image provided"}))
                continue

            # Convert the base64 string to an image
            image_data = image_data.split(",")[1]  # Remove the data URL prefix
            img_data = base64.b64decode(image_data)
            np_array = np.frombuffer(img_data, dtype=np.uint8)
            frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

            # Process the frame using YOLO model
            frame, detected_objects = process_frame(frame)

            # Combine prompts
            pre_prompt = "You are an AI assistant for a blind person. Describe the surroundings based on detected objects in a helpful and concise manner with an assumed distance and direction of objects."
            object_description = ". ".join(detected_objects)
            full_prompt = f"{pre_prompt} The detected objects are: {object_description}. User's additional context: {user_prompt}."

            print("\nPrompt to Gemini API:", full_prompt)

            # Get the response from Gemini
            content = generate_content(full_prompt)
            print("\nGemini's Response:\n", content)

            response = {
                "objects": detected_objects,
                "geminiResponse": content
            }

            # Send the response back to the client
            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        print("Client disconnected")

# Add a basic route for testing the API
@app.get("/")
def read_root():
    return {"message": "Welcome to the Anshu Gemini API"}
