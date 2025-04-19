# import pygame
# import os
# import random
# import asyncio
# import edge_tts
# from dotenv import dotenv_values
# import tempfile
# import pyttsx3

# # Load environment variables
# env_vars = dotenv_values(".env")
# InputLanguage = env_vars.get("InputLanguage", "en")  # Default to English if not set
# TTSLanguage = env_vars.get("TTSLanguage", "en")  

# # Map languages to edge_tts voices
# VOICE_MAP = {
#     "en":"en-GB-Chirp3-HD-Fenrir",
#     # English voice
#     "hi": "hi-IN-MadhurNeural"  # Hindi voice
# }

# # Initialize the TTS engine
# engine = pyttsx3.init()

# # Set a male voice
# voices = engine.getProperty('voices')
# for voice in voices:
#     if "male" in voice.name.lower():  # Look for a male voice
#         engine.setProperty('voice', voice.id)
#         break

# # Set a slower speech rate (e.g., 150 words per minute)
# engine.setProperty('rate',400)

# # Ensure the Data directory exists
# if not os.path.exists("Data"):
#     os.makedirs("Data")
    
# pygame.mixer.init()

# async def TextToAudioFile(text):
#     """Convert text to audio using edge_tts and play it directly."""
#     try:
#         voice = VOICE_MAP.get(TTSLanguage, "en-GB-Chirp3-HD-Fenrir")  # Get voice from TTSLanguage
#         communicate = edge_tts.Communicate(text, voice, pitch='+1Hz', rate='+30%')
        
#         # Create a temporary file to store the audio
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
#             temp_path = temp_file.name
#             async for chunk in communicate.stream():
#                 if chunk["type"] == "audio":
#                     temp_file.write(chunk["data"])
        
#         # Play the audio using pygame.mixer
#         print(f"Speaking: {text}")
#         pygame.mixer.music.load(temp_path)
#         pygame.mixer.music.play()

#         # Wait for the audio to finish playing
#         while pygame.mixer.music.get_busy():
#             pygame.time.Clock().tick(10)
        
#         pygame.mixer.music.stop()
#         pygame.mixer.music.unload()
        
#         # Clean up the temporary file
#         os.remove(temp_path)

#     except Exception as e:
#         print(f"Error in TTS: {e}")
#     finally:
#         try:
#             # Ensure pygame.mixer is properly stopped
#             if pygame.mixer.get_init():
#                 pygame.mixer.music.stop()
#         except Exception as e:
#             print(f"Error in finally block: {e}")
            

# def GetSpeak(text):
#     """Convert text to speech using the configured male voice and slower rate."""
#     if TTSLanguage == "hi":
#         # Use edge_tts for Hindi
#         asyncio.run(TextToAudioFile(text))
#     else:
#         # Use pyttsx3 for English
#         engine.say(text)
#         engine.runAndWait()

# def TextToSpeech(Text, func=lambda r=None: True):
#     """Convert long text to speech with a fallback message."""
#     Data = str(Text).split(".")
#     responses = {
#         "en": [
#             "The rest of the result has been printed to the chat screen, kindly check it out sir.",
#             "The rest of the text is now on the chat screen, sir, please check it.",
#             "You can see the rest of the text on the chat screen, sir.",
#             "The remaining part of the text is now on the chat screen, sir.",
#             "Sir, you'll find more text on the chat screen for you to see.",
#             "The rest of the answer is now on the chat screen, sir.",
#             "Sir, please look at the chat screen, the rest of the answer is there.",
#             "You'll find the complete answer on the chat screen, sir.",
#             "The next part of the text is on the chat screen, sir.",
#             "Sir, please check the chat screen for more information.",
#             "There's more text on the chat screen for you, sir.",
#             "Sir, take a look at the chat screen for additional text.",
#             "You'll find more to read on the chat screen, sir.",
#             "Sir, check the chat screen for the rest of the text.",
#             "The chat screen has the rest of the text, sir.",
#             "There's more to see on the chat screen, sir, please look.",
#             "Sir, the chat screen holds the continuation of the text.",
#             "You'll find the complete answer on the chat screen, kindly check it out sir.",
#             "Please review the chat screen for the rest of the text, sir.",
#             "Sir, look at the chat screen for the complete answer."
#         ],
#         "hi": [
#             "शेष परिणाम चैट स्क्रीन पर प्रदर्शित किया गया है, कृपया इसे देखें सर।",
#             "शेष पाठ अब चैट स्क्रीन पर है, सर, कृपया इसे देखें।",
#             "आप चैट स्क्रीन पर शेष पाठ देख सकते हैं, सर।",
#             "पाठ का शेष भाग अब चैट स्क्रीन पर है, सर।",
#             "सर, आपको चैट स्क्रीन पर अधिक पाठ देखने को मिलेगा।",
#             "शेष उत्तर अब चैट स्क्रीन पर है, सर।",
#             "सर, कृपया चैट स्क्रीन पर देखें, शेष उत्तर वहां है।",
#             "आपको चैट स्क्रीन पर पूरा उत्तर मिलेगा, सर।",
#             "पाठ का अगला भाग चैट स्क्रीन पर है, सर।",
#             "सर, कृपया अधिक जानकारी के लिए चैट स्क्रीन देखें।",
#             "चैट स्क्रीन पर आपके लिए और पाठ है, सर।",
#             "सर, अतिरिक्त पाठ के लिए चैट स्क्रीन देखें।",
#             "आपको चैट स्क्रीन पर और पढ़ने को मिलेगा, सर।",
#             "सर, शेष पाठ के लिए चैट स्क्रीन देखें।",
#             "चैट स्क्रीन पर शेष पाठ है, सर।",
#             "चैट स्क्रीन पर और देखने को है, सर, कृपया देखें।",
#             "सर, चैट स्क्रीन पर पाठ की निरंतरता है।",
#             "आपको चैट स्क्रीन पर पूरा उत्तर मिलेगा, कृपया इसे देखें सर।",
#             "कृपया शेष पाठ के लिए चैट स्क्रीन की समीक्षा करें, सर।",
#             "सर, पूरा उत्तर देखने के लिए चैट स्क्रीन देखें।"
#         ]
#     }

#     if len(Data) > 4 and len(Text) >= 250:
#         response = random.choice(responses.get(TTSLanguage, responses["en"]))
#         asyncio.run(TextToAudioFile(" ".join(Text.split(".")[0:2]) + ". " + response))
#     else:
#         asyncio.run(TextToAudioFile(Text))

# if __name__ == "__main__":
#     while True:
#         TextToSpeech(input("Enter the Text: "))


import os
import re
import random
import asyncio
import tempfile
import edge_tts
import pygame
import pyttsx3
from dotenv import dotenv_values
from typing import Dict, List, Optional
from dataclasses import dataclass

# Load environment variables
env_vars = dotenv_values(".env")
TTS_LANGUAGE = env_vars.get("TTSLanguage", "en").lower()
INPUT_LANGUAGE = env_vars.get("InputLanguage", "en").lower()

@dataclass
class VoiceProfile:
    voice_id: str
    rate: int = 180  # Words per minute
    pitch: int = 50   # 0-100 scale
    volume: float = 1.0

class JARVISVoice:
    def __init__(self):
        # Initialize audio system
        pygame.mixer.init()
        self._init_engines()
        self._prepare_responses()
        
        # Voice configuration
        self.voice_profiles = {
            "en": VoiceProfile(
                voice_id="en-GB-RyanNeural",
                rate=180,
                pitch=50,
                volume=1.0
            ),
            "hi": VoiceProfile(
                voice_id="hi-IN-MadhurNeural",
                rate=160,
                pitch=45,
                volume=1.0
            )
        }
        
        # Ensure data directory exists
        os.makedirs("Data", exist_ok=True)

    def _init_engines(self):
        """Initialize TTS engines with fallbacks"""
        try:
            self.local_engine = pyttsx3.init()
            voices = self.local_engine.getProperty('voices')
            for voice in voices:
                if "male" in voice.name.lower():
                    self.local_engine.setProperty('voice', voice.id)
                    break
            self.local_engine.setProperty('rate', 180)
        except Exception as e:
            print(f"Local TTS engine init failed: {e}")
            self.local_engine = None

    def _prepare_responses(self):
        """Prepare JARVIS-style responses"""
        self.responses = {
            "en": [
                "The detailed response has been displayed, sir. Please check your screen.",
                "I've provided the complete information on your display, sir.",
                "The full response is now visible on screen for your review.",
                "You'll find all the details on your display, sir.",
                "The information has been presented on screen, sir."
            ],
            "hi": [
                "पूरी जानकारी स्क्रीन पर प्रदर्शित की गई है, सर।",
                "सर, स्क्रीन पर संपूर्ण विवरण उपलब्ध है।",
                "आपके लिए सभी जानकारी स्क्रीन पर दिखाई जा रही है, सर।",
                "सर, स्क्रीन पर पूरा विवरण देख सकते हैं।",
                "सभी विवरण आपकी स्क्रीन पर प्रस्तुत किए गए हैं, सर।"
            ]
        }

    async def _edge_tts_speak(self, text: str, voice_id: str) -> bool:
        """Use edge_tts for high-quality speech synthesis"""
        try:
            communicate = edge_tts.Communicate(
                text=text,
                voice=voice_id,
                rate="+20%",
                pitch="+10Hz"
            )
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_path = tmp_file.name
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        tmp_file.write(chunk["data"])
                
                # Play audio
                self._play_audio(tmp_path)
                os.remove(tmp_path)
                return True
                
        except Exception as e:
            print(f"Edge TTS failed: {e}")
            return False

    def _play_audio(self, file_path: str):
        """Play audio file using pygame"""
        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        finally:
            pygame.mixer.music.stop()

    def _local_tts_speak(self, text: str):
        """Fallback to local TTS engine"""
        if self.local_engine:
            try:
                self.local_engine.say(text)
                self.local_engine.runAndWait()
            except Exception as e:
                print(f"Local TTS failed: {e}")

    def _should_use_edge(self, text: str) -> bool:
        """Determine if edge_tts should be used"""
        # Use edge_tts for non-English or long important phrases
        return (TTS_LANGUAGE != "en" or 
                len(text.split()) > 15 or
                any(keyword in text.lower() for keyword in ["sir", "alert", "warning"]))

    def _get_voice_profile(self) -> VoiceProfile:
        """Get appropriate voice profile"""
        return self.voice_profiles.get(TTS_LANGUAGE, self.voice_profiles["en"])

    def _process_long_text(self, text: str) -> str:
        """Handle long text with JARVIS-style interruption"""
        if len(text) > 250:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            if len(sentences) > 2:
                main_part = " ".join(sentences[:2])
                follow_up = random.choice(self.responses.get(TTS_LANGUAGE, self.responses["en"]))
                return f"{main_part}. {follow_up}"
        return text

    async def speak(self, text: str):
        """Main method to speak text with JARVIS characteristics"""
        if not text.strip():
            return

        processed_text = self._process_long_text(text)
        voice_profile = self._get_voice_profile()

        if self._should_use_edge(processed_text):
            success = await self._edge_tts_speak(processed_text, voice_profile.voice_id)
            if not success and self.local_engine:
                self._local_tts_speak(processed_text)
        elif self.local_engine:
            self._local_tts_speak(processed_text)

if __name__ == "__main__":
    jarvis_voice = JARVISVoice()