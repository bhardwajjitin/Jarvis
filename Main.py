# from Frontend.GUI import (
#     GraphicalUserInterface,
#     SetAssistantStatus,
#     ShowTextToScreen,
#     GraphicsDirectoryPath,
#     SetMicrophoneStatus,
#     AnswerModifier,
#     GetMicrophoneStatus,
#     GetAssistantStatus,
#     QueryModifier
# )
# from Backend.Model import FirstLayerDMM
# from Backend.Chatbot import Chatbot
# from Backend.Realtime import RealTimeSearchEngine
# from Backend.SpeechToText import SpeechRecognition
# from Backend.TextToSpeech import TextToSpeech
# from Backend.Automation import Automation
# from dotenv import dotenv_values
# from asyncio import run
# from time import sleep
# import os
# import subprocess
# import threading
# import json

# env_vars = dotenv_values(".env")
# Username = env_vars.get("Username")
# Assistantname = env_vars.get("Assistantname")
# DefaultMessage=f'''{Username}:Hello {Assistantname}, How are you?
# {Assistantname}:Welcome,{Username}. I am good, how can I help you?'''
# subprocess=[]
# Functions=["open","close","play","system","content","youtube search","reminder","google search"]

# def ShowDefaultChatIfNoChats():
#     File=open(r'Data\ChatLog.json','r',encoding='utf-8')
#     if len(File.read())<5:
#         with open(GraphicsDirectoryPath('Database.data'), 'w', encoding='utf-8') as file:
#             file.write("")

#         with open(GraphicsDirectoryPath('Responses.data'), 'w', encoding='utf-8') as file:
#             file.write(DefaultMessage)

# def ReadChatLogJson():
#     with open(r'Data\ChatLog.json','r',encoding='utf-8') as file:
#         chatlog_data=json.load(file)
#     return chatlog_data

# def ChatLogIntegration():
#     json_data=ReadChatLogJson()
#     formatted_chatlog=""
#     for entry in json_data:
#         if entry['role']=="User":
#             formatted_chatlog+=f"{Username}:{entry['content']}\n"
#         elif entry['role']=="Assistant":
#             formatted_chatlog+=f"{Assistantname}:{entry['content']}\n"
#     formatted_chatlog=formatted_chatlog.replace("User",Username + " ")
#     formatted_chatlog=formatted_chatlog.replace("Assistant",Assistantname + " ")

#     with open(GraphicsDirectoryPath('Database.data'), 'w', encoding='utf-8') as file:
#         file.write(AnswerModifier(formatted_chatlog))


# def ShowChatsOnGUI():
#     File=open(GraphicsDirectoryPath('Database.data'),'r',encoding='utf-8')
#     Data=File.read()
#     if len(str(Data))>0:
#         lines=Data.split("\n")
#         result='\n'.join(lines)
#         File.close()
#         File=open(GraphicsDirectoryPath('Responses.data'),'w',encoding='utf-8')
#         File.write(result)
#         File.close()


# def InitialExecution():
#     SetMicrophoneStatus("False")
#     ShowTextToScreen("")
#     ShowDefaultChatIfNoChats()
#     ChatLogIntegration()
#     ShowChatsOnGUI()

# InitialExecution()

# def MainExecution():
#     TaskExecution=False
#     ImageExecution=False
#     ImageGenerationQuery=""
#     SetAssistantStatus("Listening.....")
#     Query=SpeechRecognition()
#     ShowTextToScreen(f"{Username}:{Query}")
#     SetAssistantStatus("Thinking.....")
#     Decision=FirstLayerDMM(Query)
#     print("")
#     print(f"Decision: {Decision}")
#     print("")

#     G=any([i for i in Decision if i.startswith("general")])
#     R=any([i for i in Decision if i.startswith("realtime")])

#     Mearged_query=" and ".join([" ".join(i.split()[1:]) for i in Decision if i.startswith("general") or i.startswith("realtime")])

#     for queries in Decision:
#         if "generate " in queries:
#             ImageGenerationQuery=str(queries)
#             ImageExecution=True

#     for queries in Decision:
#         if TaskExecution==False:
#             if any(queries.startswith(func) for func in Functions):
#                 run(Automation(list(Decision)))
#                 TaskExecution=True

#     if ImageExecution==True:

#         with open(r"Frontend\Files\ImageGeneration.data","w") as file:
#             file.write(f"{ImageGenerationQuery},True")

#         try:
#             p1=subprocess.Popen(["python",r"Backend\ImageGeneration.py"],
#                                 stdout=subprocess.PIPE,
#                                 stderr=subprocess.PIPE,
#                                 stdin=subprocess.PIPE,
#                                 shell=False)
#             subprocess.append(p1)

#         except Exception as e:
#             print(f"Error in Image Generation: {e}")

#     if G and R or R:
        
#         SetAssistantStatus("Searching.....")
#         Answer=RealTimeSearchEngine(QueryModifier(Mearged_query))
#         ShowTextToScreen(f"{Assistantname}:{Answer}")
#         SetAssistantStatus("Speaking.....")
#         TextToSpeech(Answer)
#         return True
#     else:
#         for Queries in Decision:
#             if "general" in Queries:
#                 SetAssistantStatus("Thinking.....")
#                 QueryFinal=Queries.replace("general","")
#                 Answer=Chatbot(QueryModifier(QueryFinal))
#                 ShowTextToScreen(f"{Assistantname}:{Answer}")
#                 SetAssistantStatus("Answering.....")
#                 TextToSpeech(Answer)
#                 return True
#             elif "realtime" in Queries:
#                 SetAssistantStatus("Searching.....")
#                 QueryFinal=Queries.replace("realtime","")
#                 Answer=RealTimeSearchEngine(QueryModifier(QueryFinal))
#                 ShowTextToScreen(f"{Assistantname}:{Answer}")
#                 SetAssistantStatus("Answering.....")
#                 TextToSpeech(Answer)
#                 return True
#             elif "exit" in Queries:
#                 QueryFinal="Okay,Bye"
#                 Answer=Chatbot(QueryModifier(QueryFinal))
#                 ShowTextToScreen(f"{Assistantname}:{Answer}")   
#                 SetAssistantStatus("Answering.....")
#                 TextToSpeech(Answer)
#                 SetAssistantStatus("Answering.....")
#                 os._exit(1)

# def FirstThread():
#     while True:
#         CurrentStatus=GetMicrophoneStatus()
#         if CurrentStatus=="True":
#             MainExecution()

#         else:
#             AIStatus=GetAssistantStatus()

#             if "Available..." in AIStatus:
#                 sleep(0.1)

#             else:
#                 SetAssistantStatus("Available...")
                
# def SecondThread():
#     GraphicalUserInterface()

# if __name__=="__main__":
#     thread2=threading.Thread(target=FirstThread,daemon=True)
#     thread2.start()
#     SecondThread()
    
    
import random
from Backend.LLM import LLMOrchestrator
from Frontend.GUI import (
    GraphicalUserInterface,
    SetAssistantStatus,
    ShowTextToScreen,
    GraphicsDirectoryPath,
    SetMicrophoneStatus,
    AnswerModifier,
    GetMicrophoneStatus,
    GetAssistantStatus,
    QueryModifier
)
from Backend.Model import FirstLayerDMM
from Backend.Chatbot import Chatbot
from Backend.Realtime import RealtimeEngine
from Backend.SpeechToText import SpeechProcessor
from Backend.TextToSpeech import JARVISVoice
from dotenv import dotenv_values
from asyncio import run
from time import sleep
import os
import subprocess
import threading
import json
import pyaudio
import struct
from pvporcupine import Porcupine
import pygame  # For background music
import psutil  # For battery status
from datetime import datetime 
from Backend.Automation import AIControlCenter
from enum import Enum, auto # Import the new automation system
import asyncio
import subprocess
import logging
from typing import Optional
from dataclasses import dataclass
import Backend.LLM 
from Backend.LLM import LLMOrchestrator

# Initialize the AI control center


# Load environment variables
env_vars = dotenv_values(".env")
Username = env_vars.get("Username")
Assistantname = env_vars.get("Assistantname")
DefaultMessage = f'''{Username}:Hello {Assistantname}, How are you?
{Assistantname}:Welcome,{Username}. I am good, how can I help you?'''
subprocess = []


class TaskCategory(Enum):
    APPLICATION = auto()
    FILE = auto()
    MEDIA = auto()
    SYSTEM = auto()
    WEB = auto()
    PRODUCTIVITY = auto()
    HARDWARE = auto()
    AUTOMATION = auto()

FUNCTIONS = {
    # Application Control
    "open": TaskCategory.APPLICATION,
    "close": TaskCategory.APPLICATION,
    "launch": TaskCategory.APPLICATION,
    "quit": TaskCategory.APPLICATION,
    "switch": TaskCategory.APPLICATION,
    
    # File Operations
    "create file": TaskCategory.FILE,
    "delete file": TaskCategory.FILE,
    "move file": TaskCategory.FILE,
    "copy file": TaskCategory.FILE,
    "rename file": TaskCategory.FILE,
    "organize files": TaskCategory.FILE,
    "search files": TaskCategory.FILE,
    "compress": TaskCategory.FILE,
    "extract": TaskCategory.FILE,
    
    # Media Control
    "play": TaskCategory.MEDIA,
    "pause": TaskCategory.MEDIA,
    "next": TaskCategory.MEDIA,
    "volume": TaskCategory.MEDIA,
    "mute": TaskCategory.MEDIA,
    "screenshot": TaskCategory.MEDIA,
    "record screen": TaskCategory.MEDIA,
    "camera": TaskCategory.MEDIA,
    
    # System Operations
    "system": TaskCategory.SYSTEM,
    "system info": TaskCategory.SYSTEM,
    "shutdown": TaskCategory.SYSTEM,
    "restart": TaskCategory.SYSTEM,
    "sleep": TaskCategory.SYSTEM,
    "update": TaskCategory.SYSTEM,
    "cleanup": TaskCategory.SYSTEM,
    "monitor": TaskCategory.SYSTEM,
    
    # Web Operations
    "google search": TaskCategory.WEB,
    "youtube search": TaskCategory.WEB,
    "scrape": TaskCategory.WEB,
    "browse": TaskCategory.WEB,
    "download": TaskCategory.WEB,
    "post": TaskCategory.WEB,
    
    # Productivity
    "reminder": TaskCategory.PRODUCTIVITY,
    "email": TaskCategory.PRODUCTIVITY,
    "calendar": TaskCategory.PRODUCTIVITY,
    "meeting": TaskCategory.PRODUCTIVITY,
    "todo": TaskCategory.PRODUCTIVITY,
    "notes": TaskCategory.PRODUCTIVITY,
    "task": TaskCategory.PRODUCTIVITY,
    
    # Hardware Control
    "brightness": TaskCategory.HARDWARE,
    "volume": TaskCategory.HARDWARE,
    "fan speed": TaskCategory.HARDWARE,
    "rgb lights": TaskCategory.HARDWARE,
    
    # Automation
    "macro": TaskCategory.AUTOMATION,
    "record steps": TaskCategory.AUTOMATION,
    "automate": TaskCategory.AUTOMATION,
    "schedule": TaskCategory.AUTOMATION
    
}

# Enhanced function mapping to the new automation system
FUNCTION_HANDLERS = {
    TaskCategory.APPLICATION: {
        "verbs": ["open", "launch", "close", "quit", "switch"],
        "handler": "app.execute"
    },
    TaskCategory.FILE: {
        "verbs": ["create", "delete", "move", "copy", "rename", "organize"],
        "handler": "file.execute"
    },
    TaskCategory.MEDIA: {
        "verbs": ["play", "pause", "volume", "mute", "screenshot"],
        "handler": "media.execute"
    },
    TaskCategory.SYSTEM: {
        "verbs": ["shutdown", "restart", "update", "cleanup"],
        "handler": "system.execute"
    },
    TaskCategory.WEB: {
        "verbs": ["search", "browse", "download", "scrape"],
        "handler": "web.execute"
    },
    TaskCategory.PRODUCTIVITY: {
        "verbs": ["reminder", "email", "calendar", "meeting"],
        "handler": "productivity.execute"
    },
   
    TaskCategory.HARDWARE: {
        "verbs": ["brightness", "volume", "fan", "rgb"],
        "handler": "hardware.execute"
    },
    TaskCategory.AUTOMATION: {
        "verbs": ["macro", "record", "automate", "schedule"],
        "handler": "automation.execute"
    }
}

# JARVIS-style response phrases
JARVIS_PHRASES = {
    "acknowledge": [
        "Certainly, {}.",
        "Right away, {}.",
        "As you wish, {}.",
        "Processing that now, {}."
    ],
    "completed": [
        "Task completed, {}.",
        "Done, {}.",
        "Implementation successful, {}."
    ],
    "error": [
        "My apologies, {}. There seems to be an issue.",
        "Pardon me, {}. An unexpected complication arose."
    ]
}


def get_jarvis_phrase(phrase_type):
    """Return a random JARVIS-style phrase for the given context."""
    return random.choice(JARVIS_PHRASES[phrase_type]).format(Username)

def get_task_category(command: str) -> TaskCategory:
    """Determine the task category from a command"""
    command = command.lower()
    for func, category in FUNCTIONS.items():
        if func in command:
            return category
    return None

# Initialize Porcupine for wake word detection
def initialize_wake_word_detection():
    porcupine = Porcupine(
        access_key="Ua+R+88FZl1HdBkJw+UNra9+6iKYXBb6v+JsbXKZyRev+F8YxKhpdA==",  # Your Picovoice access key
        library_path="F:\\Jarvis\\libpv_porcupine.dll",  # Path to the Porcupine library
        model_path="F:\\Jarvis\\porcupine_params.pv",  # Path to the Porcupine model file
        keyword_paths=["F:\\Jarvis\\Wake-Up-My-Boy_en_windows_v3_0_0.ppn"],  # Path to the wake word file
        sensitivities=[0.5]  # Sensitivity for the wake word
    )
    return porcupine

# Initialize pygame for background music
def initialize_music():
    pygame.mixer.init()
    music_path = "F:\Jarvis\jarvis_wake_up_call.mp3"  # Path to your MP3 file
    pygame.mixer.music.load(music_path)

# # Play background music
def play_background_music():
    pygame.mixer.music.play()  # -1 means loop indefinitely

# Stop background music
def stop_background_music():
    pygame.mixer.music.stop()
    
    
# Function to listen for the wake word
def listen_for_wake_word(porcupine):
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )

    print("Listening for wake word...")
    while True:
        pcm = audio_stream.read(porcupine.frame_length)
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
        keyword_index = porcupine.process(pcm)
        if keyword_index >= 0:
            print("Wake word detected!")
            wake_up_sequence()
            audio_stream.close() # This will block until the speech is finished
            return True
        
def wake_up_sequence():
    greeting = get_greeting()
    battery_status = get_battery_status()
    JARVISVoice.speak(f"{greeting},{battery_status}")
    if("Morning Sir" in greeting):
        JARVISVoice.speak("I hope you had a good sleep. I am here to assist you with your tasks.")
        initialize_music()
        play_background_music()
    
# Show default chat if no chats exist
def ShowDefaultChatIfNoChats():
    File = open(r'Data\ChatLog.json', 'r', encoding='utf-8')
    if len(File.read()) < 5:
        with open(GraphicsDirectoryPath('Database.data'), 'w', encoding='utf-8') as file:
            file.write("")

        with open(GraphicsDirectoryPath('Responses.data'), 'w', encoding='utf-8') as file:
            file.write(DefaultMessage)

# Read chat log JSON
def ReadChatLogJson():
    with open(r'Data\ChatLog.json', 'r', encoding='utf-8') as file:
        chatlog_data = json.load(file)
    return chatlog_data

# Integrate chat log into the GUI
def ChatLogIntegration():
    json_data = ReadChatLogJson()
    formatted_chatlog = ""
    for entry in json_data:
        if entry['role'] == "User":
            formatted_chatlog += f"{Username}:{entry['content']}\n"
        elif entry['role'] == "Assistant":
            formatted_chatlog += f"{Assistantname}:{entry['content']}\n"
    formatted_chatlog = formatted_chatlog.replace("User", Username + " ")
    formatted_chatlog = formatted_chatlog.replace("Assistant", Assistantname + " ")

    with open(GraphicsDirectoryPath('Database.data'), 'w', encoding='utf-8') as file:
        file.write(AnswerModifier(formatted_chatlog))

# Show chats on the GUI
def ShowChatsOnGUI():
    File = open(GraphicsDirectoryPath('Database.data'), 'r', encoding='utf-8')
    Data = File.read()
    if len(str(Data)) > 0:
        lines = Data.split("\n")
        result = '\n'.join(lines)
        File.close()
        File = open(GraphicsDirectoryPath('Responses.data'), 'w', encoding='utf-8')
        File.write(result)
        File.close()

# Get time-based greeting
def get_greeting():
    current_hour = datetime.now().hour
    if 5 <= current_hour < 12:
        return "Good Morning Sir, Welcome Back "
    elif 12 <= current_hour < 18:
        return "Good afternoon Sir"
    else:
        return "Good evening Sir"

# Get battery status
def get_battery_status():
    battery = psutil.sensors_battery()
    if battery is None:
        return "Battery information not available."
    
    percent = battery.percent
    plugged = battery.power_plugged
    
    status = "charging" if plugged else "not charging"
    return f"Your battery is at {percent} percent and it is currently {status}."


# Ask how the user is feeling and respond accordingly
def ask_how_are_you():
    JARVISVoice.speak("How are you feeling today Sir?")
    response = SpeechProcessor().lower()
    if "good" in response or "fine" in response or "great" in response:
        JARVISVoice.speak("That's wonderful to hear!")
    elif "bad" in response or "not good" in response or "tired" in response:
        JARVISVoice.speak("I hope your day gets better!")
    else:
        JARVISVoice.speak("I'm here to help you with anything you need.")


# Initial execution setup
def InitialExecution():
    SetMicrophoneStatus("False")
    ShowTextToScreen("")
    ShowDefaultChatIfNoChats()
    ChatLogIntegration()
    ShowChatsOnGUI()

InitialExecution()

# State management
@dataclass
class AssistantState:
    current_status: str = "Sleeping..."
    last_query: str = ""
    conversation_history: list = None
    active_tasks: set = None
    
    def __post_init__(self):
        self.conversation_history = []
        self.active_tasks = set()
    
    def set_status(self, status: str):
        self.current_status = status
        # Update GUI/status display
        SetAssistantStatus(status)

# Main execution
async def MainExecution(control_center: LLMOrchestrator):
    """Enhanced main execution loop with error handling and state management"""
    state = AssistantState()
    
    while True:
        try:
            TaskExecution = False
            ImageExecution = False
            ImageGenerationQuery = ""
            
            # Listening phase
            state.set_status("Listening...")
            Query = SpeechProcessor()
            ShowTextToScreen(f"{Username}:{Query}")
            state.last_query = Query
            
            # Decision making
            state.set_status("Thinking...")
            Decision = FirstLayerDMM(Query)
            print(f"\nDecision: {Decision}\n")

            # Extract query types
            G = any(i.startswith("general") for i in Decision)
            R = any(i.startswith("realtime") for i in Decision)
            Merged_query = " and ".join(i.split(maxsplit=1)[1] for i in Decision if i.startswith(("general", "realtime")))

            # Handle image generation
            for queries in Decision:
                if "generate " in queries:
                    ImageGenerationQuery = str(queries)
                    ImageExecution = True
                    await HandleImageGeneration(ImageGenerationQuery, state)

            # Handle automation tasks
            for queries in Decision:
                if not TaskExecution and any(queries.startswith(func) for func in FUNCTIONS):
                    success, result = await control_center.execute_command(queries)
                    RespondToUser(result, success, state)
                    TaskExecution = True

            # Process general/realtime queries
            if G and R or R:
                await RealtimeEngine(Merged_query, control_center)
            else:
                tasks = []
                for Queries in Decision:
                    if "general" in Queries:
                        tasks.append(HandleGeneralQuery(Queries, control_center))
                    elif "realtime" in Queries:
                        tasks.append(handle_user_input(Queries.replace("realtime", ""), control_center))
                    elif "exit" in Queries or "sleep" in Queries:
                        await HandleExitCommand(state)
                        return False
                
                if tasks:
                    await asyncio.gather(*tasks)
                    
        except Exception as e:
            logging.error(f"MainExecution error: {e}")
            state.set_status("Error")
            await _handle_error(e, state)
            continue

async def _handle_error(error: Exception, state: AssistantState):
    """Graceful error recovery"""
    error_msg = str(error)
    if "API" in error_msg or "connection" in error_msg:
        response = "I'm having trouble connecting to services. Please check your internet connection."
    else:
        response = "Something went wrong. Let me try that again."
    
    RespondToUser(response, success=False, state=state)
    state.set_status("Recovering...")
    await asyncio.sleep(1)

# Helper functions
async def HandleImageGeneration(query: str, state: AssistantState):
    """Enhanced image generation with state tracking"""
    try:
        state.set_status("Generating image...")
        with open(r"Frontend\Files\ImageGeneration.data", "w") as file:
            file.write(f"{query},True")
        
        process = await asyncio.create_subprocess_exec(
            "python", r"Backend\ImageGeneration.py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise Exception(stderr.decode())
            
    except Exception as e:
        logging.error(f"Image Generation Error: {e}")
        await _handle_error(e, state)

def RespondToUser(response: str, success: bool = True, state: Optional[AssistantState] = None):
    """Enhanced response handling with state awareness"""
    status = "Answering..." if success else "Error..."
    if state:
        state.set_status(status)
    
    ShowTextToScreen(f"{Assistantname}:{response}")
    JARVISVoice.speak(response)
    
async def HandleGeneralQuery(query: str, control_center) -> str:
    """
    Enhanced general query handler with:
    - Proper chatbot initialization
    - Streaming support
    - Error handling
    - Conversation management
    """
    try:
        # Initialize chatbot with orchestrator
        chatbot = Chatbot(orchestrator=control_center)
        
        # Process query (automatically handles streaming if needed)
        response = await chatbot.respond(
            query=query.replace("general", "").strip(),
            stream_callback=RespondToUser  # Pass your response handler
        )
        
        return response
        
    except Exception as e:
        error_msg = f"{get_jarvis_phrase('error')} {str(e)}"
        logging.error(f"General query handling error: {e}")
        return error_msg

async def handle_user_input(prompt: str, control_center, state: AssistantState):
    """Route user input to appropriate handler"""
    realtime_engine = RealtimeEngine(control_center)
    
    def stream_handler(chunk, state):
        print(f"{Assistantname}: {chunk}", end="", flush=True)
        state.update_last_response(chunk)
    
    if is_realtime_query(prompt):
        return await realtime_engine.handle_query(
            prompt,
            stream_callback=stream_handler if len(prompt.split()) > 15 else None
        )
    else:
        # Handle general queries with Chatbot
        pass

def is_realtime_query(prompt: str) -> bool:
    """Determine if query requires real-time data"""
    triggers = ["weather", "news", "route", "stock", "email", "current", "latest"]
    return any(trigger in prompt.lower() for trigger in triggers)


async def HandleExitCommand(state: AssistantState):
    """Enhanced shutdown sequence"""
    response = "Okay, Bye"
    RespondToUser(response, state=state)
    state.set_status("Sleeping...")
    
# Thread management
async def FirstThread(control_center):
    """Enhanced wake word detection with state"""
    porcupine = initialize_wake_word_detection()
    state = AssistantState()
    while True:
        if listen_for_wake_word(porcupine):
            state.set_status("Listening...")
            if not await MainExecution(control_center):  # Uses passed orchestrator
                state.set_status("Sleeping...")
        else:
            state.set_status("Sleeping...")

    
def SecondThread():
    """GUI thread"""
    GraphicalUserInterface()

if __name__ == "__main__":
    # Initialize LLM control center
    control_center = LLMOrchestrator()
    # Warm up LLM connections
    asyncio.run(control_center.warmup_connections())
    # Start threads
    thread2 = threading.Thread(target=lambda: asyncio.run(FirstThread(control_center)), daemon=True)
    thread2.start()
    SecondThread()