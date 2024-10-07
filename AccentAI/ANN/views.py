from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from transformers import pipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import pyttsx3
import edge_tts
import asyncio
import playsound
import pandas as pd
import time
import threading
from sklearn.ensemble import GradientBoostingRegressor
import concurrent.futures


# Global variables
global start
start = time.time()

# Load dataset and train models
data = pd.read_csv('D:/Ascentt/Ritik Laptop/Vjay So/Downloads Data/ANN-Project-twelwesep/ANN-Project-twelwesep/AccentAI/ANN/static/car_dashboard_data_500_rows.csv')

fuel_tank_capacity = 50  # in liters (adjustable value)
data['Remaining Range (km)'] = (data['Fuel Level (%)'] / 100) * (fuel_tank_capacity / data['Fuel Consumption (L/100km)']) * 100

# Train Gradient Boosting model for range prediction
remaining_range_model = GradientBoostingRegressor()
remaining_range_model.fit(
    data[['Speed (km/h)', 'Throttle (%)', 'RPM', 'Fuel Level (%)', 'Fuel Consumption (L/100km)']],
    data['Remaining Range (km)']
)

# Thresholds for various metrics
engine_temp_threshold = data['Engine Temp (°C)'].quantile(0.95)
oil_pressure_threshold = data['Oil Pressure (PSI)'].quantile(0.05)
battery_voltage_threshold = data['Battery Voltage (V)'].quantile(0.05)
rpm_upper_threshold = data['RPM'].quantile(0.90)
rpm_lower_threshold = data['RPM'].quantile(0.10)

# ---------- Prediction Functions ----------
def predict_remaining_range(input_data):
    return remaining_range_model.predict([input_data])[0]

global a
a = 0

# Maintenance Alerts
def maintenance_alerts(engine_temp, oil_pressure, battery_voltage):
    alerts = []
    global a
    global start

    if engine_temp > engine_temp_threshold:
        alerts.append("High engine temperature!")
        if a == 0:
            while True:
                playsound.playsound("D:/Ascentt/Ritik Laptop/Vjay So/Downloads Data/ANN-Project-twelwesep/ANN-Project-twelwesep/AccentAI/ANN/static/output.mp3")
                if time.time() - start < 600:
                    a = 1
                else:
                    start = time.time()
                time.sleep(1)

    if oil_pressure < oil_pressure_threshold:
        alerts.append("Low oil pressure!")
    if battery_voltage < battery_voltage_threshold:
        alerts.append("Low battery voltage!")
    
    return alerts

# Brake Wear Prediction
def predict_brake_wear(brake_pressure):
    brake_pressure_threshold = data['Brake Pressure (kPa)'].mean()
    if brake_pressure > brake_pressure_threshold:
        return "Brake wear may be high. Check brakes!"
    return "Brakes are in good condition."

# Gear Shift Suggestions
def gear_shift_suggestion(gear, rpm):
    if rpm > rpm_upper_threshold and gear < 5:
        return "Consider shifting up to avoid engine strain."
    elif rpm < rpm_lower_threshold and gear > 1:
        return "Consider shifting down for optimal performance."
    else:
        return "Current gear is optimal."

# Driver Behavior Analysis
def analyze_driver_behavior(speed, throttle):
    speed_aggressive_threshold = data['Speed (km/h)'].quantile(0.90)
    throttle_aggressive_threshold = data['Throttle (%)'].quantile(0.90)
    
    if speed > speed_aggressive_threshold and throttle > throttle_aggressive_threshold:
        return "Aggressive driving detected! Consider smoother driving for better range."
    return "Driver behavior is smooth."

# ---------- Views ----------

@csrf_exempt
def predict(request):
    if request.method == "POST":
        input_data = json.loads(request.body.decode('utf-8'))  # JSON data from frontend

        remaining_range = predict_remaining_range([
            input_data['speed'],
            input_data['throttle'],
            input_data['rpm'],
            input_data['fuel_level'],
            input_data['fuel_consumption']
        ])

        maintenance_warnings = maintenance_alerts(
            input_data['engine_temp'],
            input_data['oil_pressure'],
            input_data['battery_voltage']
        )

        brake_status = predict_brake_wear(input_data['brake_pressure'])
        gear_suggestion = gear_shift_suggestion(input_data['gear'], input_data['rpm'])
        driver_behavior = analyze_driver_behavior(input_data['speed'], input_data['throttle'])

        response = {
            'fuel_efficiency': remaining_range,
            'maintenance_warnings': maintenance_warnings,
            'brake_status': brake_status,
            'gear_suggestion': gear_suggestion,
            'driver_behavior': driver_behavior
        }

        return JsonResponse(response)
    else:
        return HttpResponse("Cannot find the data")

# ---------- Voice Generation Feature ----------

async def generate_voice(text: str, Language: str):
    asyncio.sleep(1)
    voices = {
        "Bengali": "bn-IN-TanishaaNeural",
        "English": "en-IN-PrabhatNeural",
        "Gujarati": "gu-IN-NiranjanNeural",
        "Hindi": "hi-IN-MadhurNeural",
        "Kannada": "kn-IN-GaganNeural",
        "Malayalam": "ml-IN-SobhanaNeural",
        "Marathi": "mr-IN-AarohiNeural",
        "Tamil": "ta-IN-ValluvarNeural",
        "Telugu": "te-IN-MohanNeural",
    }

    try:
        tts = edge_tts.Communicate(text=text, voice=voices[Language], rate="+10%", pitch="+20Hz")
        with open("output_audio.mp3", "wb") as audio_file:
            async for chunk in tts.stream():
                if chunk["type"] == "audio":
                    audio_file.write(chunk["data"])
    except:
        pass

# ---------- Google Generative AI Integration ----------

API_KEY = 'AIzaSyAPlOApRhed-YTh3-J6iKfdu78ZTP_jQ-k'
memory = ConversationBufferMemory()
llm = ChatGoogleGenerativeAI(model="gemini-pro", api_key=API_KEY)

prompt_template = """You are a human expert with extensive knowledge in vehicle care and maintenance. Your name is 'Eco'. 
I will ask you questions about various aspects of car maintenance, such as engine troubleshooting, tire care, interior cleaning, and all other automotive issues. 
Please provide detailed, practical, and human-like advice as if you were performing the task yourself or advising a customer based on personal experience.
Question: {promp}"""

prompt1 = """Detect the language of the following text and return only the name of the language without any additional information:
{}"""

prompt2 = """Convert the following text into Target Language without any additional information.
Text: '{}'
Target Language: '{}'"""

prompt3 = """Translate the following text from Source Language to Target Language. Return only the translated text without any additional information:
Text: "{}"
Source Language: "{}"
Target Language: "{}"""

new_prompt="""You are Eco an car care expert who is able to solve everything and provide a solution to the problem.
            The problem I am facing is that {prompt}. Detect the language of the prompt and provide a detailed solution to it in that language only.
            Provide the solution in paragraph form without using any bullet points, stars, or special characters.Please make sure that you are detecting
            and translating and providing correctly.
            Dont return question what i asked just start providing solution"""


chain = LLMChain(llm=llm, memory=memory, prompt=PromptTemplate(template=new_prompt))

def detect_language(transcript):
    return llm.invoke(prompt1.format(transcript)).content

def process_text(transcript):
    result = chain.invoke({"prompt": transcript})
    return result["text"]

# ---------- Views for Index and Processing Transcripts ----------

def index(request):
    return render(request, 'index.html')

@csrf_exempt
def process_transcript(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        transcript = data.get('transcript')

        if transcript.strip() != "":
            with concurrent.futures.ThreadPoolExecutor() as executor:
    # Schedule both tasks to run in parallel
                future_language = executor.submit(detect_language, transcript)
                future_result = executor.submit(process_text, transcript)

                # Wait for both tasks to complete and get their results
                language = future_language.result()
                result = future_result.result()
                print("Language Detected: ", language)
                # text = llm.invoke(prompt3.format(transcript, language, "English")).content
                # print("Converted Text: ", text)
                # result = chain.invoke({"prompt": transcript})
                
                # print("Result in English: ", result)
                # final_result = llm.invoke(prompt2.format(result, language)).content
                print("Result in Native Language: ", result)

            asyncio.run(generate_voice(result.replace("*", ""), Language=language))
            playsound.playsound("D:/Ascentt/Ritik Laptop/Vjay So/Downloads Data/ANN-Project-twelwesep/ANN-Project-twelwesep/AccentAI/output_audio.mp3", True)

            return JsonResponse({'status': 'success', 'transcript': transcript})
        
    return JsonResponse({'status': 'failed', 'message': 'Invalid request method'})













# from django.shortcuts import render
# from django.http import JsonResponse,HttpResponse
# from django.views.decorators.csrf import csrf_exempt
# import json
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.memory import ConversationBufferMemory
# from transformers import pipeline
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# import pyttsx3
# import edge_tts
# import asyncio
# import playsound
# import requests
# import pandas as pd
# import threading
# import time
# from sklearn.ensemble import GradientBoostingRegressor
# global start
# start=time.time()
# # model= pipeline(model="huseinzol05/text-to-speech-tacotron-male")

# data = pd.read_csv('D:\Ascentt\Ritik Laptop\Vjay So\Downloads Data\ANN-Project-twelwesep\ANN-Project-twelwesep\AccentAI\ANN\static\car_dashboard_data_500_rows.csv')

# # Assuming a fixed fuel tank capacity for the car
# fuel_tank_capacity = 50  # in liters (you can adjust this value)

# # Add a 'Remaining Range (km)' column to the dataset
# # Formula: Remaining Range (km) = (Fuel Level (%) / 100) * (Fuel Tank Capacity / Fuel Consumption) * 100
# data['Remaining Range (km)'] = (data['Fuel Level (%)'] / 100) * (fuel_tank_capacity / data['Fuel Consumption (L/100km)']) * 100

# # Train a model for Remaining Range prediction
# remaining_range_model = GradientBoostingRegressor()
# remaining_range_model.fit(
#     data[['Speed (km/h)', 'Throttle (%)', 'RPM', 'Fuel Level (%)', 'Fuel Consumption (L/100km)']],
#     data['Remaining Range (km)']
# ) 

# # Thresholds
# engine_temp_threshold = data['Engine Temp (°C)'].quantile(0.95)
# oil_pressure_threshold = data['Oil Pressure (PSI)'].quantile(0.05)
# battery_voltage_threshold = data['Battery Voltage (V)'].quantile(0.05)
# rpm_upper_threshold = data['RPM'].quantile(0.90)
# rpm_lower_threshold = data['RPM'].quantile(0.10)

# # --------- Prediction Functions ---------
# # Remaining Range Prediction using Gradient Boosting
# def predict_remaining_range(input_data):
#     return remaining_range_model.predict([input_data])[0]

# global a
# a=0
# # Maintenance Alerts
# def maintenance_alerts(engine_temp, oil_pressure, battery_voltage):
#     alerts = []
#     global a
#     global start
#     if engine_temp > engine_temp_threshold:
#         alerts.append("High engine temperature!")
#         while a==0:
#             playsound.playsound("D:/Ascentt/Ritik Laptop/Vjay So/Downloads Data/ANN-Project-twelwesep/ANN-Project-twelwesep/AccentAI/ANN/static/output.mp3")
#             if time.time()-start<600:
#                 a=1
#             else:
#                 start=time.time()
#             time.sleep(1)
#     if oil_pressure < oil_pressure_threshold:
#         alerts.append("Low oil pressure!")
#     if battery_voltage < battery_voltage_threshold:
#         alerts.append("Low battery voltage!")
#     return alerts

# # Brake Wear Prediction
# def predict_brake_wear(brake_pressure):
#     brake_pressure_threshold = data['Brake Pressure (kPa)'].mean()
#     if brake_pressure > brake_pressure_threshold:
#         return "Brake wear may be high. Check brakes!"
#     return "Brakes are in good condition."

# # Gear Shift Suggestions
# def gear_shift_suggestion(gear, rpm):
#     if rpm > rpm_upper_threshold and gear < 5:
#         return "Consider shifting up to avoid engine strain."
#     elif rpm < rpm_lower_threshold and gear > 1:
#         return "Consider shifting down for optimal performance."
#     else:
#         return "Current gear is optimal."

# # Driver Behavior Analysis
# def analyze_driver_behavior(speed, throttle):
#     speed_aggressive_threshold = data['Speed (km/h)'].quantile(0.90)
#     throttle_aggressive_threshold = data['Throttle (%)'].quantile(0.90)
    
#     if speed > speed_aggressive_threshold and throttle > throttle_aggressive_threshold:
#         return "Aggressive driving detected! Consider smoother driving for better range."
#     return "Driver behavior is smooth."



# @csrf_exempt
# def predict(request):
#     if request.method=="POST":
#         input_data = json.loads(request.body.decode('utf-8'))  # JSON data from frontend

#         remaining_range = predict_remaining_range([
#             input_data['speed'],
#             input_data['throttle'],
#             input_data['rpm'],
#             input_data['fuel_level'],
#             input_data['fuel_consumption']
#         ])

#         maintenance_warnings = maintenance_alerts(
#             input_data['engine_temp'],
#             input_data['oil_pressure'],
#             input_data['battery_voltage']
#         )

#         brake_status = predict_brake_wear(input_data['brake_pressure'])
#         gear_suggestion = gear_shift_suggestion(input_data['gear'], input_data['rpm'])
#         driver_behavior = analyze_driver_behavior(input_data['speed'], input_data['throttle'])

#         response = {
#             'fuel_efficiency': remaining_range,
#             'maintenance_warnings': maintenance_warnings,
#             'brake_status': brake_status,
#             'gear_suggestion': gear_suggestion,
#             'driver_behavior': driver_behavior
#         }

#         return JsonResponse(response)
#     else:
#         return HttpResponse("Can not find the data")



# async def generate_voice(text: str,Language: str):

#     asyncio.sleep(1)
#     voices={
#         "Bengali":"bn-IN-TanishaaNeural",
#         "English":"en-IN-PrabhatNeural",
#         "Gujarati":"gu-IN-NiranjanNeural",
#         "Hindi":"hi-IN-MadhurNeural",
#         "Kannada":"kn-IN-GaganNeural",
#         "Malayalam":"ml-IN-SobhanaNeural",
#         "Marathi":"mr-IN-AarohiNeural",
#         "Tamil":"ta-IN-ValluvarNeural",
#         "Telugu":"te-IN-MohanNeural",}
#     try:
#         tts = edge_tts.Communicate(text=text, voice=voices[Language], rate="+10%", pitch="+20Hz")
#         with open("output_audio.mp3", "wb") as audio_file:
#             async for chunk in tts.stream():
#                 if chunk["type"] == "audio":
#                     audio_file.write(chunk["data"])
#     except:
#         pass

# API_KEY = 'AIzaSyAPlOApRhed-YTh3-J6iKfdu78ZTP_jQ-k'


# memory=ConversationBufferMemory()


# llm=ChatGoogleGenerativeAI(model="gemini-pro",api_key="AIzaSyAPlOApRhed-YTh3-J6iKfdu78ZTP_jQ-k")

# prompt="""You are a human expert with extensive knowledge in vehicle care and maintenance your name is 'Eco'. 
# I will ask you questions about various aspects of car maintenance, such as engine troubleshooting, tyre care, interior cleaning, and all other automotive issues. 
# Please provide detailed, practical, and human-like advice as if you were performing the task yourself or advising a customer based on personal experience.

# If I ask a question outside of vehicle-related topics, politely inform me that your expertise is limited to vehicle care and maintenance.

# Question: {promp}"""

# prompt1="""Detect the language of the following text and return only the name of the language without any additional information:

# {}"""

# prompt2="""Convert the following text into Target Language without any additional information.

# Text: '{}'
# Target Language: '{}'"""

# prompt3="""Translate the following text from Source Language to Target Language. Return only the translated text without any additional information:

# Text: "{}"
# Source Language: "{}"
# Target Language: "{}"""
# prompt=PromptTemplate(template=prompt)
# chain=LLMChain(llm=llm,memory=memory,prompt=prompt)



# def index(request):
#     return render(request, 'index.html')
 
# @csrf_exempt
# def process_transcript(request):
#     if request.method == 'POST':
#         data = json.loads(request.body.decode('utf-8'))
#         transcript = data.get('transcript')
#         if transcript.strip()!="":
#             language=llm.invoke(prompt1.format(transcript)).content
#             print("Language Detected: ",language)
#             text=llm.invoke(prompt3.format(transcript,language,"English")).content
#             print("Converted Text: ",text)
#             result=chain.invoke({"promp":text})
#             result=result["text"]
#             print("Result in English: ",result)
#             final_result=llm.invoke(prompt2.format(result,language)).content
#             print("result in Native: ",final_result)
#             asyncio.run(generate_voice(final_result.replace("*",""),Language=language))
#             playsound.playsound("D:/Ascentt/Ritik Laptop/Vjay So/Downloads Data/ANN-Project-twelwesep/ANN-Project-twelwesep/AccentAI/output_audio.mp3",True)
#                 # model(result.content)
#             return JsonResponse({'status': 'success', 'transcript': transcript})
        
#     return JsonResponse({'status': 'failed', 'message': 'Invalid request method'})
