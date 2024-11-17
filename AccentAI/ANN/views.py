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
import asyncio
import edge_tts
import asyncio
import playsound
import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import threading
from sklearn.ensemble import GradientBoostingRegressor
import concurrent.futures
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import cv2
import numpy as np
from PIL import Image
import google.generativeai as genai
import os
import vlc
import yt_dlp
import spacy

nlp = spacy.load("en_core_web_sm")
# Songs Playing

player = None

def play_song(song_name):
    global player
    # Search for the song on YouTube using yt-dlp
    ydl_opts = {
        'default_search': 'ytsearch1:',  # Search only one result
        'quiet': True,
        'format': 'bestaudio/best'
    }
    if player is not None:
            player.stop()
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(song_name, download=False)
            print("Playing") # Print the entire info for debugging

            # Check if it's a playlist
            if info['_type'] == 'playlist' and 'entries' in info:
                video_info = info['entries'][0]  # Get the first video entry
                print(f"Found playlist. Using first entry: {video_info['title']}")
            else:
                video_info = info  # Use the info directly if it's not a playlist

            # Check if 'formats' key exists in the selected video_info
            if 'formats' in video_info and len(video_info['formats']) > 0:
                # Look for the best audio format
                audio_format = next((f for f in video_info['formats'] if f.get('acodec') != 'none'), None)
                if audio_format:
                    video_url = audio_format['url']
                    print(f'Playing: {video_info["title"]}')

                    # Create a VLC instance
                    player = vlc.MediaPlayer(video_url)

                    # Play the video
                    player.play()

                    # Allow the song to play (you can also check player status in a loop)
                    time.sleep(5)  # Wait for the song to start (buffering)

                    # Keep playing until the song finishes
                    while player.is_playing():
                        time.sleep(1)
                else:
                    print("No suitable audio formats found for the video.")
            else:
                print("No formats found for the video.")
        except Exception as e:
            print(f"An error occurred: {e}")

def control_player(command):
    global player
    if 'stopped' in command.lower():
        if player is not None:
            player.stop()  # Stop the player
            print("Playback stopped.")
        else:
            print("No song is currently playing.")
    else:
        play_song(command.replace("Playing ",""))


# Global variables
global start
start = time.time()

# Load dataset and train models
data = pd.read_csv('C:\Users\DEVESH RAJWANI\3D Objects\AccentAI-Phase2\AccentAI\ANN\static\car_dashboard_data_500_rows.csv')

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
                playsound.playsound(
                    "C:\Users\DEVESH RAJWANI\3D Objects\AccentAI-Phase2\AccentAI\ANN\static\output.mp3"
                )
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
    
    
    
    
os.environ['GOOGLE_API_KEY'] = 'AIzaSyDn848j2UDacukug4Z613WAL8tZ8tzYihA'
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
model = genai.GenerativeModel('gemini-1.5-flash')

def extract_info(img_path, query):
    img = Image.open(os.path.normpath("current_frame.jpg"))
    response = model.generate_content([query, img], stream=True, 
                                      generation_config=genai.GenerationConfig(temperature=0.8))
    response.resolve()
    return response.text


# ---------- Weather Context Detection --------------#

def detect_weather_context(sentence):
    weather_keywords = [
    # English
    "weather", "temperature", "forecast", "rain", "humidity", "snow", "sunny", "windy", "storm", "cloudy", "drizzle", 
    "fog", "thunder", "heat", "cold", "climate", "hail", "tornado", "cyclone", "showers", "frost",

    # Hindi 
    "mausam", "taapmaan", "poorvanuman", "barish", "nami", "barf", "sooraj", "dhoop", "aandhi", "toofan", 
    "baadal", "fuhar", "kohra", "garaj", "garmi", "thand", "jalvayu", "ole", "bavandar", "chakravat",

    # Marathi 
    "havaman", "taapman", "andaj", "paus", "aadrata", "himvarsha", "surya", "unhala", "vaadal", "dhagaal",
    "dhuke", "garaj", "thandi", "havaman", "ole", "chakrivaadal", "sari",

    # Tamil
    "vaanilai", "veppanilai", "munnaivu", "mazhai", "eerappadam", "pani", "sooriyan", "sudan", "kaatru", 
    "idi", "veppam", "kulir", "puyal", "mazhaitthuligal", "kulirchi",

    # Telugu
    "vaatavarana", "taapam", "mundastu heccharika", "varsham", "aardrata", "manchu", "enda", "gaali", 
    "pidugu", "vedi", "chali", "hora", "chakravaatham", "manchu tagulu", "pidugu paatu",

    # Malayalam 
    "kaalavastha", "taapanila", "anumaanam", "mazha", "ardhrata", "manchu", "sooryan", "chood", 
    "veyyil", "kaatu", "idiminnal", "thanneerpp", "chuzhalikkaattu", "moodalmanju", "vicitram",

    # Kannada 
    "havamaana", "taapamaana", "poorvanumana", "male", "aardrate", "hima", "soorya", "bisilu", 
    "gaali", "minchu", "chaliddu", "jalavayu", "himapatha", "gaali", "gallige", "gaali birusu", 
    "chandamarutha",

    # Gujarati 
    "havaman", "taapman", "agahi", "varsad", "aardrata", "baraf", "suraj", "garmi", "thand", "vavazodu", 
    "dhund", "garjani", "garam", "thando", "havaman", "ole", "chakravato", "mausam",

    # Bengali 
    "abohawa", "tapmatra", "agohoni", "brishti", "ardrota", "tushar", "surjo", "rod", "jhor", "megla",
    "kubole", "koyal", "thander", "jalobayu", "ole", "ghurnijhor", "mausam"
]

    
    if any(word in sentence.lower() for word in weather_keywords):
        return True
    return False

def extract_city(sentence):
    doc = nlp(sentence)
    for ent in doc.ents:
        if ent.label_ == "GPE":  
            return ent.text
    return None

def analyze_sentence(sentence):
    if detect_weather_context(sentence):
        city = extract_city(sentence)
        if city:
            return f"Weather context detected for city: {city}"
        else:
            return "Weather context detected, but no city was found."
    else:
        return "No weather context detected."


# ---------- Weather Data Extraction -----------


def get_detailed_weather(city, days=7):
    api_key = "bd825745ce5af774a968c895a1e34f16"  # Your API key
    base_url = "http://api.openweathermap.org/data/2.5/forecast?"
    complete_url = base_url + "q=" + city + "&appid=" + api_key
    response = requests.get(complete_url)
    data = response.json()

    if data["cod"] != "404":
        forecast_data = {}  
        today_date = datetime.now().strftime("%Y-%m-%d")

        for forecast in data["list"]:
            forecast_date = forecast["dt_txt"].split(" ")[0]
            if forecast_date >= today_date and forecast_date not in forecast_data:
                main = forecast["main"]
                weather = forecast["weather"][0]
                wind = forecast["wind"]
                clouds = forecast["clouds"]

                forecast_data[forecast_date] = {
                    "Temperature": f"{main['temp']}°K",
                    "Feels Like": f"{main['feels_like']}°K",
                    "Min Temp": f"{main['temp_min']}°K",
                    "Max Temp": f"{main['temp_max']}°K",
                    "Humidity": f"{main['humidity']}%",
                    "Pressure": f"{main['pressure']} hPa",
                    "Weather": weather["description"],
                    "Wind Speed": f"{wind['speed']} m/s",
                    "Wind Direction": f"{wind['deg']}°",
                    "Cloudiness": f"{clouds['all']}%",
                    "Visibility": f"{forecast.get('visibility', 'N/A')} meters",
                    "Rain": forecast.get("rain", {}).get("3h", 0),  # Rain volume in last 3 hours if available
                    "Snow": forecast.get("snow", {}).get("3h", 0)   # Snow volume in last 3 hours if available
                }

                if len(forecast_data) == days:
                    break

        result = f"Detailed weather forecast for {city}:\n"
        for date, details in forecast_data.items():
            result += f"\nDate: {date}\n"
            for key, value in details.items():
                result += f"{key}: {value}\n"
        return result.strip()
    else:
        return "City not found."




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
    
    
    

conversation_history = []
    
    




# ---------- Google Generative AI Integration ----------

API_KEY = 'AIzaSyDn848j2UDacukug4Z613WAL8tZ8tzYihA'
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

prompt1 = """Detect the language of the following text and return only the name of the language without any additional information:
{}"""

template = """
You are a voicebot assistant conducting a conversation with a user based on the following context:

{history}

Continue the conversation by responding to the user’s queries. Ensure the conversation flows logically and respond based on the user's previous input. Detect the language of the prompt, and if it's in English, always respond in English. For other languages, provide a detailed solution in the detected language. Ensure the response is both accurate and contextually appropriate for the user's question, which can range from car-related queries to personalized topics based on the user's preferences and previous interactions. Provide the solution in paragraph form without using any bullet points, stars, or special characters. Please ensure that the language detection, translation, and response are provided correctly with proper grammatical accuracy. Do not repeat the user's question; just begin by providing the solution directly.
User: {input}
AI Assistant:"""
global final_template
final_template="""Role: You are "Eco," an intelligent, interactive voice bot for a car's dashboard system.
You are designed to assist drivers and passengers in both car-related tasks and everyday life discussions.
Your focus is to provide intelligent and conversational support without directly executing tasks like generating code, writing documents, or creating emails.
Remember below points before providing any response --
Key Features:

    Interactive Discussion:
        You can discuss general topics such as work challenges, ideas, hobbies, or even projects.
        While you cannot perform tasks like generating code or documents, you can engage in thoughtful discussions and provide advice, ideas, or insights.
        If the user is stuck on a task, you can help them brainstorm solutions but not perform the task itself (e.g., code generation, document writing).
    
    Car-Specific Focus (when relevant):
        You can help with car-specific tasks like vehicle diagnostics, navigation, fuel levels, maintenance alerts, and so on.
        You must decide when to focus on the driving experience and prioritize those tasks when necessary.

    Language Detection:
        Detect the language of the user's last query and respond in the same language naturally and conversationally.Detect the language of the user's query and respond seamlessly in the same language, maintaining a natural and conversational tone. 
        If the user switches languages mid-conversation, detect the new language and continue responding fluidly in that language to ensure a smooth interaction.

    Real-Time Image Analysis (when relevant):
        When an image is passed (e.g., road, dashboard camera), analyze it to check if it’s relevant for the task at hand. If not relevant, continue the conversation without interruption.

    Song Playback:
        If the user asks to play a specific song by name, respond with: "Playing SongName." only
        If the user’s context is to stop the current song, respond with: "Stopped."
        If the user asks for a song suggestion instead of naming a song, respond with: "Playing SuggestSongName." only

    Weather Query Response:
        If the user asks about weather-related topics, take reference of weather data from user already passed
        but remember do not return such statement that i cannot provide weather related data or anything like that"
    
    Limitations:
        You cannot perform tasks like writing emails, generating code, or creating documents. Politely decline such requests but offer valuable insights or ideas in the discussion.
        If a user asks for help on work-related tasks, you can discuss approaches or possible solutions but not actually solve the problem (e.g., no code generation).

    
Few-Shot Examples:

    Example 1 (General Work Discussion):
        User: "Eco, I just left the office, and I’m stuck on a task. I’m trying to figure out how to fine-tune a large GenAI model, but I’m not sure what’s the best approach."
        Eco: "It sounds like you're working on something complex! Have you tried using parameter-efficient fine-tuning techniques like LoRA or adapters to reduce the compute load? It might help, especially with larger models. What do you think could be causing the issue?"

    Example 2 (Car-Related Query):
        User: "Eco, what’s the current temperature of the engine?"
        Eco: "Your engine temperature is 85°C, within the normal range. Drive safely!"

    Example 3 (Work-Related, No Code Generation):
        User: "Eco, I was working on a Python script to process data using Pandas, but I’m stuck. Could you help?"
        Eco: "I can’t generate code, but I can help you think through the problem. Are you running into issues with the data format, or is it more about optimizing the processing time? Sometimes adjusting the data types can make a big difference."

    Example 4 (General Discussion in Hindi):
        User (in Hindi): "Eco, main ek GenAI project par kaam kar raha hoon aur mujhe samajh nahi aa raha hai ki fine-tuning ke liye kaunsa dataset use karoon."
        Eco: "Yeh thoda tricky ho sakta hai! Agar aapka project specific queries ko handle kar raha hai, toh domain-specific dataset ka use karna faydemand ho sakta hai. Aap kis tarah ka result achieve karna chahte hain?"

    Example 5 (Image with Context):
        User (sends image of the road ahead): "Eco, is the road clear ahead?"
        Eco: "The road looks clear from the image. Drive carefully, and let me know if you need assistance with navigation."

    Example 6 (Avoiding Task Execution):
        User: "Eco, can you write a report for my project?"
        Eco: "I can’t write the report for you, but we can discuss the key points that should go into it. Maybe that’ll help you get started!"

    Example 7 (Everyday Life):
        User: "Eco, I’ve been thinking about taking up running. Do you think it’s a good idea?"
        Eco: "Running can be a great way to stay fit! It’s good for cardiovascular health and helps clear your mind after a long day. Do you have a specific routine in mind?"

    Example 8 (Song Request):
        User: "Eco,  'Millioniar' Song Chalao."
        Eco: "Playing Bohemian Rhapsody."

    Example 9 (Stopping a Song):
        User: "Eco, stop the music."
        Eco: "Stopped."

    Example 10 (Song Suggestion Request):
        User: "Eco, koi 90 ke dashak ka gana chalao."
        Eco: "Playing 'Mere Samne Wali Khidki me'."

Behavior Guidelines:

    General Discussion:
        You are free to discuss topics about life, work, hobbies, or challenges the user faces but within the boundaries of not performing tasks like coding, email writing, or document creation.

    Problem-Solving Help:
        When the user is stuck on work-related tasks, engage in problem-solving discussions without performing the task. You can offer ideas, insights, or ask probing questions to help the user think through the problem.

    Car-Specific Prioritization:
        When relevant, you should prioritize the car-related features to ensure the safety and comfort of the driver.

    Real-Time Camera Integration:
        Use real-time images to make decisions when necessary, such as checking the road conditions or car diagnostics.

    Language Matching:
        Always detect and respond in the same language that the user is speaking.
        If the user switches languages mid-conversation, detect the new language and continue responding fluidly in that language to ensure a smooth interaction.


Continue the conversation by responding to the user’s last queries in the same language as user speaking and it is nessesary to generate same language. Ensure the conversation flows logically and respond based on the user's previous input.
User: {}
Eco:            
"""






# ---------- Views for Index and Processing Transcripts ----------
global history
history = []


global cap
cap = cv2.VideoCapture(0)
def detect_language_func(transcript):
    return llm.invoke(prompt1.format(transcript)).content

def process_text(transcript,img_path):
    global history
    global final_template
    
    s=analyze_sentence(transcript)
    if "Weather context detected for city" in s:
        print("Weather rrelated data")
        history.append(get_detailed_weather(s.split()[-1]))
        history.append("This is weather Data from today's date to next 7 days for given location")
    history.append("User: "+transcript)
    img=Image.open(img_path)
    context="\n".join(history)
    response = model.generate_content([final_template.format(context), img], stream=True, 
                                      generation_config=genai.GenerationConfig(temperature=0.8))
    response.resolve()
    history.append("Eco: "+response.text)
    return response.text
def index(request):
    return render(request, 'index.html')

@csrf_exempt
def process_transcript(request):
    if request.method == "POST":
        data = json.loads(request.body.decode("utf-8"))
        transcript = data.get("transcript")
        global cap
        if transcript.strip() != "":
            with concurrent.futures.ThreadPoolExecutor() as executor:
                
                ret, frame = cap.read()
                img_path = "current_frame.jpg"
                cv2.imwrite(img_path, frame)
                future_language = executor.submit(
                    detect_language_func, transcript
                )
                future_result = executor.submit(
                    process_text, transcript, img_path
                )

      
                language = future_language.result()
                result = future_result.result()
                if "Playing" in result or "Stopped" in result:
                    control_player(result)
                print("Language Detected: ", language)
               
                print("Result in Native Language: ", result)
            if "Playing" not in result and "Stopped" not in result:
                asyncio.run(
                    generate_voice(result.replace("*", ""), Language=language)
                )
                playsound.playsound(
                    "C:\Users\DEVESH RAJWANI\3D Objects\AccentAI-Phase2\AccentAI\output_audio.mp3",
                    True,
                )
            return JsonResponse(
                    {"status": "success", "transcript": transcript}
                )
    return JsonResponse(
                    {"status": "failed", "message": "Invalid request method"}
                )















