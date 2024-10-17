import cv2
from PIL import Image
import google.generativeai as genai
import os
 
# Configure Google Gemini LLM API
os.environ['GOOGLE_API_KEY'] = 'AIzaSyAkZSrqr11htt4WKPwVMq3LZf_qPukI7E4'
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
 
# Load the model (assuming 'gemini-1.5-flash' is available)
model = genai.GenerativeModel('gemini-1.5-flash')
 
# Global variable to store conversation context
conversation_history = []
 
# Function to interact with LLM and maintain conversation context
def interact_with_llm(query, img_path=None):
    global conversation_history
 
    # If an image is provided, load it and send it with the query
    if img_path:
        img = Image.open(img_path)
        input_data = [query, img]
    else:
        input_data = [query]
   
    # Add the query to conversation history for context
    conversation_history.append(query)
 
    # Build the context for the current interaction
    context = "\n".join(conversation_history)
 
    # Generate the response using LLM with conversation history
    response = model.generate_content([context], stream=True,
                                      generation_config=genai.GenerationConfig(temperature=0.8))
    response.resolve()
 
    # Add the response to the conversation history
    conversation_history.append(response.text)
 
    return response.text
 
# Start video capture
cap = cv2.VideoCapture(0)
 
while True:
    ret, frame = cap.read()
    if not ret:
        break
 
    # Show the video feed
    cv2.imshow('Live Feed', frame)
 
    # Assuming 'transcript' is where the voice input is being stored
    # Replace with actual logic to capture voice commands
    transcript = input("Voice Input: ")
 
    # Check if the command involves extracting info from the frame
    if "describe" in transcript:
        # Save the current frame as an image
        img_path = "current_frame.jpg"
        cv2.imwrite(img_path, frame)
 
        # Query LLM with the current frame
        query = "Describe the object in the image."
        result = interact_with_llm(query, img_path)
 
        # Display the response from the LLM
        print(f"LLM Response: {result}")
 
    elif "exit" in transcript:
        # Exit if the command is "exit"
        break
 
    # Press 'ESC' to exit the video stream
    if cv2.waitKey(1) & 0xFF == 27:
        break
 
# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()