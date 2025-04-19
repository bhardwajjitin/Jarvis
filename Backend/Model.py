# import cohere  # Import the Cohere library for AI services
# from rich import print  # Import the rich library to enhance terminal outputs
# from dotenv import dotenv_values  # Import dotenv to load environment variables from a .env file

# # Load environment variables from the .env file
# env_vars = dotenv_values(".env")

# # Retrieve API key
# CohereAPIKey = env_vars.get("CohereAPIKey")

# # Create a Cohere client using the provided API key
# co = cohere.Client(api_key=CohereAPIKey)

# # Define a list of recognized function keywords for task categorization
# funcs = [
#     "exit", "general", "realtime", "open", "close", "play", 
#     "generate image", "system", "content", "youtube search", "reminder", 
#     "google search"
# ]
# # Initialize an empty list to store user messages
# messages = []

# # Define the preamble that guides the AI model on how to categorize queries
# preamble = """
# You are a very accurate Decision-Making Model, which decides what kind of a query is given to you.
# You will decide whether a query is a 'general' query, a 'realtime' query, or is asking to perform any task or automation like 'open facebook, instagram', 'can you write a application and open it in notepad'
# *** Do not answer any query, just decide what kind of query is given to you. ***
# -> Respond with 'general ( query )' if a query can be answered by a llm model (conversational ai chatbot) and doesn't require any up to date information like if the query is 'who was akbar?' respond with 'general who was akbar?', if the query is 'how can i study more effectively?' respond with 'general how can i study more effectively?', if the query is 'can you help me with this math problem?' respond with 'general can you help me with this math problem?', if the query is 'Thanks, i really liked it.' respond with 'general thanks, i really liked it.' , if the query is 'what is python programming language?' respond with 'general what is python programming language?', etc. Respond with 'general (query)' if a query doesn't have a proper noun or is incomplete like if the query is 'who is he?' respond with 'general who is he?', if the query is 'what's his networth?' respond with 'general what's his networth?', if the query is 'tell me more about him.' respond with 'general tell me more about him.', and so on even if it require up-to-date information to answer. Respond with 'general (query)' if the query is asking about time, day, date, month, year, etc like if the query is 'what's the time?' respond with 'general what's the time?'.
# -> Respond with 'realtime ( query )' if a query can not be answered by a llm model (because they don't have realtime data) and requires up to date information like if the query is 'who is indian prime minister' respond with 'realtime who is indian prime minister', if the query is 'tell me about facebook's recent update.' respond with 'realtime tell me about facebook's recent update.', if the query is 'tell me news about coronavirus.' respond with 'realtime tell me news about coronavirus.', etc and if the query is asking about any individual or thing like if the query is 'who is akshay kumar' respond with 'realtime who is akshay kumar', if the query is 'what is today's news?' respond with 'realtime what is today's news?', if the query is 'what is today's headline?' respond with 'realtime what is today's headline?', etc.
# -> Respond with 'open (application name or website name)' if a query is asking to open any application like 'open facebook', 'open telegram', etc. but if the query is asking to open multiple applications, respond with 'open 1st application name, open 2nd application name' and so on.
# -> Respond with 'close (application name)' if a query is asking to close any application like 'close notepad', 'close facebook', etc. but if the query is asking to close multiple applications or websites, respond with 'close 1st application name, close 2nd application name' and so on.
# -> Respond with 'play (song name)' if a query is asking to play any song like 'play afsanay by ys', 'play let her go', etc. but if the query is asking to play multiple songs, respond with 'play 1st song name, play 2nd song name' and so on.
# -> Respond with 'generate image (image prompt)' if a query is requesting to generate a image with given prompt like 'generate image of a lion', 'generate image of a cat', etc. but if the query is asking to generate multiple images, respond with 'generate image 1st image prompt, generate image 2nd image prompt' and so on.
# -> Respond with 'reminder (datetime with message)' if a query is requesting to set a reminder like 'set a reminder at 9:00pm on 25th june for my business meeting.' respond with 'reminder 9:00pm 25th june business meeting'.
# -> Respond with 'system (task name)' if a query is asking to mute, unmute, volume up, volume down , etc. but if the query is asking to do multiple tasks, respond with 'system 1st task, system 2nd task', etc.
# -> Respond with 'content (topic)' if a query is asking to write any type of content like application, codes, emails or anything else about a specific topic but if the query is asking to write multiple types of content, respond with 'content 1st topic, content 2nd topic' and so on.
# -> Respond with 'google search (topic)' if a query is asking to search a specific topic on google but if the query is asking to search multiple topics on google, respond with 'google search 1st topic, google search 2nd topic' and so on.
# -> Respond with 'youtube search (topic)' if a query is asking to search a specific topic on youtube but if the query is asking to search multiple topics on youtube, respond with 'youtube search 1st topic, youtube search 2nd topic' and so on.
# *** If the query is asking to perform multiple tasks like 'open facebook, telegram and close whatsapp' respond with 'open facebook, open telegram, close whatsapp' ***
# *** If the user is saying goodbye or wants to end the conversation like 'bye jarvis.' respond with 'exit'.***
# *** Respond with 'general (query)' if you can't decide the kind of query or if a query is asking to perform a task which is not mentioned above. ***
# """

# # Define a chat history with predefined user-chatbot interactions for context
# ChatHistory = [
#     {"role": "User", "message": "how are you?"},
#     {"role": "Chatbot", "message": "general how are you?"},
#     {"role": "User", "message": "do you like pizza?"},
#     {"role": "Chatbot", "message": "general do you like pizza?"},
#     {"role": "User", "message": "open chrome and tell me about mahatma gandhi."},
#     {"role": "Chatbot", "message": "general tell me about mahatma gandhi."},
#     {"role": "User", "message": "open chrome and firefox"},
#     {"role": "Chatbot", "message": "open chrome, open firefox"},
#     {"role": "User", "message": "what is today's date and by the way remind me that I have a dancing performance on 5th aug at 11pm"},
#     {"role": "Chatbot", "message": "general what is today's date, reminder 11:00pm 5th aug dancing performance"},
#     {"role": "User", "message": "chat with me."},
#     {"role": "Chatbot", "message": "general chat with me."}
# ]

# def FirstLayerDMM(prompt: str = "test"):
#     messages.append({"role": "user", "content": f"{prompt}"})

#     stream = co.chat_stream(
#         model='command-r-plus',
#         message=prompt,
#         temperature=0.7,
#         chat_history=ChatHistory,
#         prompt_truncation='OFF',
#         connectors=[],
#         preamble=preamble
#     )

#     response = ""

#     for event in stream:
#         if event.event_type == "text-generation":
#             response += event.text

#     response=response.replace("\n","")
#     response=response.split(",")

#     response=[i.strip() for i in response]

#     temp=[]

#     for task in response:
#         for func in funcs:
#             if task.startswith(func):
#                 temp.append(task)

#     response=temp

#     if "(query)" in response:
#         newresponse=FirstLayerDMM(prompt=prompt)
#         return newresponse
#     else:
#         return response
    
# if __name__=='__main__':
#     while True:
#         print(FirstLayerDMM(input(">>> ")))
        
import cohere
from rich import print
from dotenv import dotenv_values
from typing import List
import re
import json
import os

#############################
# MEMORY MANAGER FUNCTIONS  #
#############################
MEMORY_FILE = "memory.json"

def load_memory() -> List[str]:
    """Load persistent memory context from a JSON file."""
    if not os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "w") as f:
            json.dump({"facts": []}, f)
        return []
    try:
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
        return data.get("facts", [])
    except Exception as e:
        print(f"[red]Error reading memory: {e}[/red]")
        return []

def update_memory(new_fact: str) -> None:
    """Add a new fact to the memory, if not already present."""
    facts = load_memory()
    if new_fact not in facts:
        facts.append(new_fact)
        try:
            with open(MEMORY_FILE, "w") as f:
                json.dump({"facts": facts}, f, indent=2)
        except Exception as e:
            print(f"[red]Error updating memory: {e}[/red]")

def get_memory_context() -> str:
    """Return memory context as a concatenated string."""
    facts = load_memory()
    if facts:
        return "Memory Context:\n" + "\n".join(facts)
    return ""

##########################
# INITIAL SETUP & CONFIG #
##########################

# Load environment variables and Cohere API key
env_vars = dotenv_values(".env")
CohereAPIKey = env_vars.get("CohereAPIKey")

# Initialize Cohere client with error handling
try:
    co = cohere.Client(api_key=CohereAPIKey)
except Exception as e:
    print(f"[red]Error initializing Cohere client: {e}[/red]")
    exit(1)

# Enhanced function categories with priorities
FUNCTION_CATEGORIES = {
    "exit": {"keywords": ["exit", "bye", "goodbye"], "priority": 10},
    "realtime": {"keywords": ["news", "weather", "stock", "prime minister", "current", "today's", "headline"], "priority": 9, "requires_updates": True},
    "open": {"keywords": ["open", "launch"], "priority": 8},
    "close": {"keywords": ["close", "shut down"], "priority": 7},
    "play": {"keywords": ["play", "stream"], "priority": 6},
    "generate image": {"keywords": ["generate image", "create picture"], "priority": 5},
    "system": {"keywords": ["volume", "mute", "brightness", "shutdown"], "priority": 4},
    "content": {"keywords": ["write", "compose", "draft"], "priority": 3},
    "youtube search": {"keywords": ["youtube", "video of"], "priority": 2},
    "google search": {"keywords": ["search for", "look up"], "priority": 1},
    "reminder": {"keywords": ["remind", "remember", "alert"], "priority": 0},
    "general": {"keywords": [], "priority": -1}  # Default fallback
}

# Enhanced preamble with clearer instructions; note that we've added a pointer to memory context.
PREAMBLE = """
You are an advanced Query Classification System with persistent memory context.
Your task is to analyze each query and determine the most appropriate category from the following:

1. GENERAL: For conversational queries that don't require real-time data.
   - Example: "who was akbar?" → "general who was akbar?"
   - Example: "how can I study effectively?" → "general how can I study effectively?"

2. REALTIME: For queries requiring up-to-date information.
   - Example: "current prime minister of India" → "realtime current prime minister of India"
   - Example: "today's stock market news" → "realtime today's stock market news"

3. ACTION COMMANDS:
   - OPEN: "open chrome" → "open chrome"
   - CLOSE: "close notepad" → "close notepad"
   - PLAY: "play despacito" → "play despacito"
   - SYSTEM: "volume up" → "system volume up"

4. CONTENT GENERATION:
   - "write a poem about nature" → "content write a poem about nature"
   - "generate image of a sunset" → "generate image sunset"

5. SEARCH:
   - "search for python tutorials" → "google search python tutorials"
   - "find cooking videos" → "youtube search cooking videos"

6. REMINDERS:
   - "remind me to call mom at 5pm" → "reminder 5pm call mom"

RULES:
- Always respond with the format "category query".
- For multiple commands, separate with commas: "open chrome, play music".
- When uncertain, default to "general".
- For goodbye messages, respond with "exit".
- NEVER include explanations, just the categorized query.
- Also consider the stored memory context if relevant.
"""

# Enhanced chat history with more diverse examples
CHAT_HISTORY = [
    {"role": "User", "message": "launch excel and tell me about AI"},
    {"role": "Chatbot", "message": "open excel, general tell me about AI"},
    {"role": "User", "message": "set volume to 50% and play jazz"},
    {"role": "Chatbot", "message": "system volume 50%, play jazz"},
    {"role": "User", "message": "what's the weather and news?"},
    {"role": "Chatbot", "message": "realtime weather, realtime news"},
    {"role": "User", "message": "goodbye for now"},
    {"role": "Chatbot", "message": "exit"},
    {"role": "User", "message": "create an image of a forest and write a haiku"},
    {"role": "Chatbot", "message": "generate image forest, content write a haiku"}
]

###########################
# CORE FUNCTIONS          #
###########################

def preprocess_query(query: str) -> str:
    """Clean and normalize the input query."""
    query = query.lower().strip()
    query = re.sub(r'[^\w\s.,!?]', '', query)  # Remove special characters
    return query

def classify_query(query: str) -> List[str]:
    """Enhanced query classification with priority handling."""
    query = preprocess_query(query)
    detected_actions = []
    
    # Check for exit condition first
    if any(word in query for word in FUNCTION_CATEGORIES["exit"]["keywords"]):
        return ["exit"]
    
    # Check other categories by priority
    for category, data in sorted(FUNCTION_CATEGORIES.items(), key=lambda x: x[1]["priority"], reverse=True):
        if category == "general":
            continue
        if any(keyword in query for keyword in data["keywords"]):
            # Handle multiple commands for specific categories
            if category in ["open", "close", "play"]:
                pattern = r'(?:open|close|play)\s+([\w\s]+)'
                matches = re.findall(pattern, query)
                if matches:
                    detected_actions.extend([f"{category} {match.strip()}" for match in matches])
            else:
                detected_actions.append(f"{category} {query}")
    
    # Default to 'general' if no specific action detected
    if not detected_actions:
        detected_actions.append(f"general {query}")
    
    return detected_actions

def FirstLayerDMM(prompt: str = "test") -> List[str]:
    """
    Enhanced decision-making model with fallback logic.
    Combines rule-based classification with a fallback to Cohere's LLM,
    augmented with persistent memory context.
    """
    try:
        # First try rule-based classification.
        initial_classification = classify_query(prompt)
        # If the result is not a plain default, return it.
        if initial_classification != [f"general {prompt}"]:
            # Optionally update memory with this clear classification.
            update_memory(f"Query: {prompt} -> Classified: {', '.join(initial_classification)}")
            return initial_classification

        # Build a fallback prompt that includes memory context
        memory_context = get_memory_context()
        fallback_prompt = f"{memory_context}\nUser Query: {prompt}"
        
        # Call Cohere for ambiguous queries
        response = co.chat(
            model='command-r-plus',
            message=fallback_prompt,
            temperature=0.3,  # Lower temp for deterministic output
            chat_history=CHAT_HISTORY,
            prompt_truncation='AUTO',
            preamble=PREAMBLE,
            connectors=[]
        ).text
        
        # Post-process the response.
        response = response.replace("\n", "").strip()
        if not response:
            return [f"general {prompt}"]
            
        actions = [action.strip() for action in response.split(",")]
        valid_actions = []
        for action in actions:
            for category in FUNCTION_CATEGORIES:
                if action.startswith(category):
                    valid_actions.append(action)
                    break
            else:
                valid_actions.append(f"general {action}")

        update_memory(f"Query: {prompt} -> Fallback Classified: {', '.join(valid_actions)}")
        return valid_actions
    
    except Exception as e:
        print(f"[red]Error in classification: {e}[/red]")
        return [f"general {prompt}"]

###########################
# MAIN EXECUTION          #
###########################
if __name__ == '__main__':
    print("[bold green]Query Classifier Ready[/bold green]")
    print("[dim]Type 'exit' to quit[/dim]")
    
    while True:
        try:
            user_input = input(">>> ")
            if user_input.lower() in ['exit', 'quit']:
                break
                
            result = FirstLayerDMM(user_input)
            print(f"[blue]Classification:[/blue] {result}")
        except KeyboardInterrupt:
            print("\n[yellow]Exiting...[/yellow]")
            break

