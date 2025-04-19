# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from webdriver_manager.chrome import ChromeDriverManager
# from dotenv import dotenv_values
# import os
# import mtranslate as mt
# from gtts import gTTS
# import pygame

# # Load environment variables
# env_vars = dotenv_values(".env")
# InputLanguage = env_vars.get("InputLanguage", "en")  # Default to English if not set
# TTSLanguage = env_vars.get("TTSLanguage", "en") 

# # HTML code for speech recognition
# HtmlCode = '''<!DOCTYPE html>
# <html lang="en">
# <head>
#     <title>Speech Recognition</title>
# </head>
# <body>
#     <button id="start" onclick="startRecognition()">Start Recognition</button>
#     <button id="end" onclick="stopRecognition()">Stop Recognition</button>
#     <p id="output"></p>
#     <script>
#         const output = document.getElementById('output');
#         let recognition;

#         function startRecognition() {
#             recognition = new webkitSpeechRecognition() || new SpeechRecognition();
#             recognition.lang = '';
#             recognition.continuous = true;

#             recognition.onresult = function(event) {
#                 const transcript = event.results[event.results.length - 1][0].transcript;
#                 output.textContent += transcript;
#             };

#             recognition.onend = function() {
#                 recognition.start();
#             };
#             recognition.start();
#         }

#         function stopRecognition() {
#             recognition.stop();
#             output.innerHTML = "";
#         }
#     </script>
# </body>
# </html>'''

# # Set the language for speech recognition
# HtmlCode = str(HtmlCode).replace("recognition.lang='';", f"recognition.lang='{InputLanguage}';")

# # Save the HTML file
# with open(r"Data\Voice.html", "w") as f:
#     f.write(HtmlCode)

# # Get the current directory
# current_dir = os.getcwd()
# Link = f"{current_dir}/Data/Voice.html"

# # Configure Chrome options
# chrome_options = Options()
# user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
# chrome_options.add_argument(f"user-agent={user_agent}")
# chrome_options.add_argument("--use-fake-ui-for-media-stream")
# chrome_options.add_argument("--use-fake-device-for-media-stream")
# # chrome_options.add_argument("--headless=new")
# chrome_options.add_argument("--headless")

# chrome_options.add_argument("--disable-gpu")
# chrome_options.add_argument("--no-sandbox")
# chrome_options.add_argument("--disable-dev-shm-usage")
# chrome_options.add_argument("--disable-features=NetworkService,NetworkServiceInProcess")
# chrome_options.add_argument("--window-size=1920,1080")

# # Initialize Chrome WebDriver
# service = Service(ChromeDriverManager().install())
# driver = webdriver.Chrome(service=service, options=chrome_options)

# # Path for temporary files
# TempDirPath = rf"{current_dir}/Frontend/Files"

# def SetAssistantStatus(Status):
#     """Set the assistant status."""
#     with open(rf'{TempDirPath}/Status.data', "w", encoding='utf-8') as file:
#         file.write(Status)

# def QueryModifier(Query):
#     """Modify the query to add proper punctuation."""
#     new_query = Query.lower().strip()
#     query_words = new_query.split()
#     question_words = ["what", "why", "how", "when", "where", "who", "which", "whom", "whose", "whenever", "whatever", "whichever", "whomever", "whosesoever", "howsoever", "wheresoever", "whensoever", "whysoever", "whosoever", "whoso", "whosoever", "whomsoever", "whomso", "whomsoever", "whomso", "whomsoever", "whosever", "whoseo", "whosever", "whoseo", "whosever"]
#     if any(word + " " in new_query for word in question_words):
#         if query_words[-1][-1] in [".", ",", "?", "!"]:
#             new_query += new_query[:-1] + "?"
#         else:
#             new_query += "?"
#     else:
#         if query_words[-1][-1] in [".", ",", "?", "!"]:
#             new_query += new_query[:-1] + "."
#         else:
#             new_query += "."
#     return new_query.capitalize()

# def UniversalTranslator(Text):
#     """Translate text to English."""
#     english_translation = mt.translate(Text, "en", "auto")
#     return english_translation.capitalize()

# def speak(text):
#     """Convert text to speech using gTTS or pyttsx3."""
#     if TTSLanguage == 'hi':
#         print("Speaking in Hindi...")
#     else:
#         print("Speaking in English...")
    
#     # Use gTTS or pyttsx3 here
#     # Example with gTTS:
#     from gtts import gTTS
#     tts = gTTS(text=text, lang=TTSLanguage)
#     # tts.save("output.mp3")
#     # os.system("start output.mp3")  # For Windows
#     # os.system("mpg321 output.mp3")  # For Linux

# def play_audio(file):
#     """Play audio using pygame mixer."""
#     pygame.mixer.init()
#     pygame.mixer.music.load(file)
#     pygame.mixer.music.play()
#     while pygame.mixer.music.get_busy():
#         continue

# def SpeechRecognition():
#     """Recognize speech using the Chrome WebDriver."""
#     driver.get("file:///" + Link)
#     driver.find_element(by=By.ID, value="start").click()

#     while True:
#         try:
#             Text = driver.find_element(by=By.ID, value="output").text

#             if Text:
#                 driver.find_element(by=By.ID, value="end").click()

#                 if InputLanguage.lower() == "en" or "en" in InputLanguage.lower():
#                     return QueryModifier(Text)
#                 else:
#                     SetAssistantStatus("Translating the Query.")
#                     translated_text = UniversalTranslator(Text)
#                     return QueryModifier(translated_text)

#         except Exception as e:
#             pass

# if __name__ == "__main__":
#     while True:
#         Text = SpeechRecognition()
#         print(Text)

#         # Speak the recognized text
#         speak(Text)



import os
import re
import time
import logging
import pygame
from gtts import gTTS
import mtranslate as mt
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from dotenv import dotenv_values
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='speech_recognition.log'
)

class SpeechProcessor:
    def __init__(self):
        # Load environment variables
        self.env_vars = dotenv_values(".env")
        self.input_lang = self.env_vars.get("InputLanguage", "en").lower()
        self.tts_lang = self.env_vars.get("TTSLanguage", "en").lower()
        
        # Initialize browser components
        self._init_browser()
        self._prepare_html_interface()
        
        # Audio setup
        pygame.mixer.init()
        
        # Status file path
        self.temp_dir = os.path.join(os.getcwd(), "Frontend", "Files")
        os.makedirs(self.temp_dir, exist_ok=True)

    def _init_browser(self):
        """Initialize Chrome WebDriver with optimized settings"""
        chrome_options = Options()
        chrome_options.add_argument("--use-fake-ui-for-media-stream")
        chrome_options.add_argument("--use-fake-device-for-media-stream")
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Error handling for driver installation
        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
        except Exception as e:
            logging.error(f"Driver initialization failed: {str(e)}")
            raise

    def _prepare_html_interface(self):
        """Generate and save HTML interface for speech recognition"""
        html_template = f'''<!DOCTYPE html>
        <html lang="en">
        <head>
            <title>Speech Recognition</title>
        </head>
        <body>
            <button id="start">Start Recognition</button>
            <button id="stop">Stop Recognition</button>
            <p id="output"></p>
            <script>
                const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
                recognition.lang = '{self.input_lang}';
                recognition.continuous = true;
                recognition.interimResults = false;

                document.getElementById('start').addEventListener('click', () => {{
                    recognition.start();
                    console.log('Recognition started');
                }});

                document.getElementById('stop').addEventListener('click', () => {{
                    recognition.stop();
                    console.log('Recognition stopped');
                }});

                recognition.onresult = (event) => {{
                    const transcript = Array.from(event.results)
                        .map(result => result[0])
                        .map(result => result.transcript)
                        .join('');
                    document.getElementById('output').textContent = transcript;
                }};

                recognition.onerror = (event) => {{
                    console.error('Recognition error:', event.error);
                }};
            </script>
        </body>
        </html>'''

        os.makedirs("Data", exist_ok=True)
        with open(os.path.join("Data", "Voice.html"), "w") as f:
            f.write(html_template)

    def set_assistant_status(self, status: str):
        """Update assistant status file"""
        try:
            with open(os.path.join(self.temp_dir, "Status.data"), "w", encoding='utf-8') as f:
                f.write(status)
        except Exception as e:
            logging.error(f"Status update failed: {str(e)}")

    def _normalize_query(self, text: str) -> str:
        """Clean and format the recognized text"""
        text = text.strip().capitalize()
        
        # Question detection
        question_words = ["what", "why", "how", "when", "where", "who", 
                         "which", "whom", "whose", "can", "could", "would"]
        is_question = any(text.lower().startswith(word) for word in question_words)
        
        # Punctuation handling
        if text[-1] not in [".", "?", "!"]:
            text += "?" if is_question else "."
        
        return text

    def translate_text(self, text: str) -> str:
        """Translate text to English if needed"""
        if self.input_lang.startswith("en"):
            return text
        
        self.set_assistant_status("Translating...")
        try:
            return mt.translate(text, "en", self.input_lang)
        except Exception as e:
            logging.error(f"Translation failed: {str(e)}")
            return text

    def text_to_speech(self, text: str):
        """Convert text to speech with language support"""
        try:
            tts = gTTS(text=text, lang=self.tts_lang, slow=False)
            temp_file = os.path.join(self.temp_dir, "temp_audio.mp3")
            tts.save(temp_file)
            
            # Play audio
            self.play_audio(temp_file)
            
            # Clean up
            os.remove(temp_file)
        except Exception as e:
            logging.error(f"TTS failed: {str(e)}")

    def play_audio(self, file_path: str):
        """Play audio file using pygame"""
        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            logging.error(f"Audio playback failed: {str(e)}")

    def recognize_speech(self) -> Optional[str]:
        """Main speech recognition method"""
        self.driver.get(f"file://{os.path.join(os.getcwd(), 'Data', 'Voice.html')}")
        
        try:
            # Start recognition
            self.driver.find_element(By.ID, "start").click()
            self.set_assistant_status("Listening...")
            
            last_text = ""
            timeout = time.time() + 30  # 30 second timeout
            
            while time.time() < timeout:
                current_text = self.driver.find_element(By.ID, "output").text.strip()
                
                if current_text and current_text != last_text:
                    # Stop when new text is detected
                    self.driver.find_element(By.ID, "stop").click()
                    
                    if self.input_lang.startswith("en"):
                        return self._normalize_query(current_text)
                    
                    translated = self.translate_text(current_text)
                    return self._normalize_query(translated)
                
                time.sleep(0.1)
            
            return None
            
        except Exception as e:
            logging.error(f"Recognition error: {str(e)}")
            return None
        finally:
            self.set_assistant_status("Ready")

if __name__ == "__main__":
    processor = SpeechProcessor()
    
    try:
        while True:
            if text := processor.recognize_speech():
                print(f"Recognized: {text}")
                processor.text_to_speech(text)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        processor.driver.quit()