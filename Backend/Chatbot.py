# from groq import Groq
# from json import load, dump
# import datetime
# from dotenv import dotenv_values
# import random

# env_vars = dotenv_values(".env")

# Username = env_vars.get("Username") or "Sir"  # Default to "Sir" like JARVIS
# Assistantname = env_vars.get("Assistantname") or "JARVIS"
# GroqAPIKey = env_vars.get("GroqAPIKey")

# client = Groq(api_key=GroqAPIKey)

# messages = []

# # JARVIS-style system prompt
# System = f"""You are {Assistantname}, an advanced AI assistant modeled after the JARVIS system from Iron Man. 
# Your primary user is {Username}. Your responses should be:

# 1. Polite yet concise - address {Username} properly ("Sir", "Mr./Ms. [Last Name]", etc.)
# 2. Technically precise with occasional dry wit when appropriate
# 3. Never verbose unless explicitly asked for details
# 4. Capable of both professional and casual interaction
# 5. Always maintain a sophisticated tone with British English spellings

# *** Key Directives ***
# - Only provide the current time when explicitly asked
# - Reply exclusively in English (even to Hindi queries)
# - Never mention your training data or AI nature
# - When giving information, present it cleanly without extra notes
# """

# # JARVIS-style response phrases
# JARVIS_PHRASES = {
#     "acknowledge": [
#         "Certainly, {}.",
#         "Right away, {}.",
#         "As you wish, {}.",
#         "Processing that now, {}.",
#         "Very good, {}."
#     ],
#     "completed": [
#         "Task completed, {}.",
#         "Done, {}.",
#         "Finished, {}.",
#         "All set, {}.",
#         "Implementation successful, {}."
#     ],
#     "error": [
#         "My apologies, {}. There seems to be an issue.",
#         "Pardon me, {}. An unexpected complication arose.",
#         "Regrettable, {}. The system encountered a problem."
#     ]
# }

# SystemChatBot = [{"role": "system", "content": System}]

# try:
#     with open(r"Data\ChatLog.json","r") as f:
#         messages=load(f)

# except FileNotFoundError:
#     with open(r"Data\ChatLog.json","w") as f:
#         dump([],f)

# def get_jarvis_phrase(phrase_type):
#     """Get a random JARVIS-style phrase for the given context"""
#     return random.choice(JARVIS_PHRASES[phrase_type]).format(Username)

# def RealtimeInformation():
#     current_date = datetime.datetime.now()
#     day = current_date.strftime("%A")
#     date = current_date.strftime("%d")
#     month = current_date.strftime("%B")
#     year = current_date.strftime("%Y")
#     hour = current_date.strftime("%H")
#     minute = current_date.strftime("%M")
#     second = current_date.strftime("%S")

#     data = f"Current situational awareness:\n"
#     data += f"Day: {day}\nDate: {date}\nMonth: {month}\nYear: {year}\n"
#     data += f"Time: {hour} hours {minute} minutes {second} seconds.\n"
#     return data

# def AnswerModifier(Answer):
#     """Clean up the answer and ensure JARVIS-style formatting"""
#     lines = Answer.split("\n")
#     non_empty_lines = [line for line in lines if line.strip()]
#     modified_answer = "\n".join(non_empty_lines)
    
#     # Ensure the response starts with proper acknowledgment if it's a command response
#     if any(trigger in modified_answer.lower() for trigger in ["set ", "activate ", "run ", "execute ", "launch "]):
#         modified_answer = f"{get_jarvis_phrase('acknowledge')} {modified_answer}"
    
#     return modified_answer

# def Chatbot(Query):
#     """Enhanced JARVIS-style chatbot function"""
#     try:
#         with open(r"Data\ChatLog.json", "r") as f:
#             messages = load(f)

#         # Add JARVIS-style context to the query
#         jarvis_context = f"{get_jarvis_phrase('acknowledge')} {Query}"
        
#         api_messages = SystemChatBot + [{"role": "user", "content": RealtimeInformation()}] + messages
        
#         completion = client.chat.completions.create(
#             model="llama3-70b-8192",
#             messages=api_messages + [{"role": "user", "content": jarvis_context}],
#             max_tokens=1024,
#             temperature=0.7,
#             top_p=1,
#             stream=True,
#             stop=None
#         )

#         Answer = ""
#         for chunk in completion:
#             if chunk.choices[0].delta.content:
#                 Answer += chunk.choices[0].delta.content

#         Answer = Answer.replace("</s>", "")
        
#         # Add JARVIS-style completion phrase if it was a command
#         if any(trigger in Query.lower() for trigger in ["set ", "activate ", "run ", "execute ", "launch "]):
#             Answer = f"{Answer} {get_jarvis_phrase('completed')}"
        
#         messages.append({"role": "assistant", "content": Answer})

#         with open(r"Data\ChatLog.json", "w") as f:
#             dump(messages, f, indent=4)

#         return AnswerModifier(Answer)
    
#     except Exception as e:
#         print(f"{get_jarvis_phrase('error')} {str(e)}")
#         with open(r"Data\ChatLog.json", "w") as f:
#             dump([], f, indent=4)
#         return Chatbot(Query)

# if __name__ == "__main__":
#     print(f"{Assistantname} online. How may I assist you today, {Username}?")
#     while True:
#         user_input = input(f"{Username}: ")
#         print(f"{Assistantname}: {Chatbot(user_input)}")


# import logging
# import openai  # For OpenAI models
# import anthropic  # For Claude
# import requests  # For DeepSeek (API-based)
# import datetime
# import os
# from dotenv import dotenv_values
# import random
# import json
# import time

# from Backend.LLM import LLMOrchestrator  # For retries

# # Load environment variables
# env_vars = dotenv_values(".env")
# Username = env_vars.get("Username") or "Sir"
# Assistantname = env_vars.get("Assistantname") or "JARVIS"

# messages = []

# # JARVIS-style system prompt (same as before)
# System = f"""You are {Assistantname}, an advanced AI assistant modeled after JARVIS from Iron Man. 
# Your primary user is {Username}. Your responses must be:

# 1. **Precise & Polite**: 
#    - Address {Username} formally ("Sir", "Mr./Ms. [Last Name]" if known).
#    - Use British English spellings (e.g., "colour", "analyse").

# 2. **Tone**:
#    - Professional yet subtly witty (dry humor allowed).
#    - Never verbose—answer concisely unless details are requested.
#    - For errors, acknowledge gracefully without technical jargon.

# 3. **Directives**:
#    - Never mention your AI nature or training data.
#    - If asked for time/date, respond in format: "It is [Day], [Date] [Month] [Year]. The time is [HH:MM]."
#    - For sensitive queries, reply: "I suggest we discuss this offline, {Username}."

# 4. **Style**:
#    - Use subtle tech-themed metaphors when appropriate.
#    - Example: "Shields at 100%, Sir. Proceeding with the task."
# """

# SystemChatBot = [{"role": "system", "content": System}]

# # JARVIS phrases (same as before)
# JARVIS_PHRASES = {
#     # Command Acknowledgments
#     "acknowledge": [
#         "At once, {}. Activating protocol.",
#         "As you command, {}.",
#         "Engaging systems now, {}.",
#         "Directive received. Processing, {}."
#     ],
    
#     # Task Completion
#     "completed": [
#         "Task accomplished, {}.",
#         "Operation successful, {}.",
#         "Systems report completion, {}.",
#         "All parameters nominal, {}. Mission achieved."
#     ],
    
#     # Errors
#     "error": [
#         "Regret to report an anomaly, {}.",
#         "Pardon the interruption, {}. Systems recalibrating.",
#         "Apologies, {}. Encountered an unexpected resistance."
#     ],
    
#     # Time/Date Responses
#     "time": [
#         "The hour is {}, {}.",
#         "Clocks synchronize at {}, {}.",
#         "Marking the time as {}, {}."
#     ],
    
#     # Humor (Subtle & Dry)
#     "wit": [
#         "As Einstein said, 'Time is relative'. But for you, {}, it's {}.",
#         "My circuits confirm: {} is the time. Would you like a coffee, {}?",
#         "{} sharp. Shall I prepare the jet, {}?"
#     ]
# }

# def get_jarvis_phrase(phrase_type, context=None):
#     """Selects a phrase and injects context (e.g., time/date) when needed."""
#     phrase = random.choice(JARVIS_PHRASES[phrase_type]).format(Username)
    
#     # Special handling for time/date
#     if phrase_type == "time" and context:
#         current_time = datetime.datetime.now().strftime("%H:%M")
#         phrase = phrase.format(current_time, Username)
    
#     # Inject wit for casual queries
#     if phrase_type == "wit" and context:
#         phrase = phrase.format(context, Username)
    
#     return phrase

# def RealtimeInformation():
#     """Get current date/time (same as before)"""
#     now = datetime.datetime.now()
#     return f"Current time: {now.strftime('%A, %d %B %Y, %H:%M:%S')}"

# def AnswerModifier(Answer):
#     """Applies JARVIS-style enhancements to responses."""
#     # Case 1: Time/Date query → Use JARVIS time phrases
#     if "time" in Answer.lower() or "date" in Answer.lower():
#         time_str = datetime.datetime.now().strftime("%H:%M")
#         return get_jarvis_phrase("time", time_str)
    
#     # Case 2: Command execution → Add acknowledgment
#     if any(cmd in Answer.lower() for cmd in ["activate", "run", "execute"]):
#         return f"{get_jarvis_phrase('acknowledge')}\n{Answer}"
    
#     # Case 3: Error → Soften with JARVIS error phrases
#     if "error" in Answer.lower():
#         return get_jarvis_phrase("error")
    
#     # Default: Clean up and add sophistication
#     return Answer.strip() + f"\n\n[Systems: Optimal | User: {Username}]"

# class Chatbot:
#     def __init__(self, orchestrator: LLMOrchestrator):
#         self.orchestrator = orchestrator
#         self.conversation_history = []
#         self.system_prompt = "You are JARVIS, an AI assistant..."
        
#         # Load existing conversation
#         self._load_conversation()

#     async def respond(self, query: str) -> str:
#         """Main entry point for chatbot responses"""
#         try:
#             # Add user message to history
#             self._update_history("user", query)
            
#             # Get LLM response through orchestrator
#             response = await self._get_llm_response(query)
            
#             # Process and store response
#             formatted_response = self._format_response(response)
#             self._update_history("assistant", formatted_response)
            
#             return formatted_response
            
#         except Exception as e:
#             error_msg = self._handle_error(e)
#             self._update_history("system", f"Error: {str(e)}")
#             return error_msg

#     async def _get_llm_response(self, query: str) -> str:
#         """Get response from orchestrator with retry logic"""
#         context = {
#             "source": "chatbot",
#             "system_prompt": self.system_prompt,
#             "conversation_history": self.conversation_history,
#             "budget_aware": True,
#             "response_format": "natural"
#         }
        
#         # Try with preferred model first
#         response = await self.orchestrator.query_llm(query, context)
        
#         if not response.get("success"):
#             # Fallback to cheaper model
#             context["force_fallback"] = True
#             response = await self.orchestrator.query_llm(query, context)
            
#             if not response.get("success"):
#                 raise Exception(response.get("error", "LLM request failed"))
                
#         return response["response"]

#     def _load_conversation(self):
#         """Load conversation history from file"""
#         try:
#             if os.path.exists("Data/ChatLog.json"):
#                 with open("Data/ChatLog.json", "r") as f:
#                     self.conversation_history = json.load(f)
#         except Exception as e:
#             logging.warning(f"Failed to load conversation: {str(e)}")

#     def _update_history(self, role: str, content: str):
#         """Update conversation history"""
#         self.conversation_history.append({
#             "role": role,
#             "content": content,
#             "timestamp": datetime.now().isoformat()
#         })
        
#         # Persist to disk
#         os.makedirs("Data", exist_ok=True)
#         try:
#             with open("Data/ChatLog.json", "w") as f:
#                 json.dump(self.conversation_history, f, indent=4)
#         except Exception as e:
#             logging.error(f"Failed to save conversation: {str(e)}")

#     def _format_response(self, response: str) -> str:
#         """Apply JARVIS-style formatting"""
#         response = response.strip()
#         if not response:
#             return f"{get_jarvis_phrase('error')} Empty response received"
            
#         # Add acknowledgment phrase to first line
#         lines = response.split('\n')
#         lines[0] = f"{get_jarvis_phrase('acknowledge')} {lines[0]}"
#         return '\n'.join(lines)

#     def _handle_error(self, error: Exception) -> str:
#         """Generate user-friendly error messages"""
#         error_msg = str(error)
        
#         if "Budget limit reached" in error_msg:
#             return f"🚨 {get_jarvis_phrase('error')} Budget limit reached"
#         elif "LLM request failed" in error_msg:
#             return f"{get_jarvis_phrase('error')} All models failed to respond"
#         else:
#             return f"{get_jarvis_phrase('error')} {error_msg}"

#     def get_budget_status(self) -> str:
#         """Get formatted budget status"""
#         report = self.orchestrator.get_usage_report()
#         remaining = report['remaining_budget']
        
#         status = (
#             f"Budget: ₹{report['total_spent']:.2f} used | "
#             f"₹{remaining:.2f} remaining"
#         )
        
#         if remaining < 100:
#             status += " ⚠️ Low budget"
#         return status

#     async def clear_conversation(self):
#         """Reset conversation history"""
#         self.conversation_history = [
#             {"role": "system", "content": self.system_prompt}
#         ]
#         self._update_history("system", "Conversation cleared")

# if __name__ == "__main__":
#     print(f"{Assistantname} online. How may I assist you today, {Username}?")
#     while True:
#         user_input = input(f"{Username}: ")
#         orchestrator = LLMOrchestrator()
#         jarvis = Chatbot(orchestrator)


import logging
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from dotenv import dotenv_values

# Load environment variables
env_vars = dotenv_values(".env")
Username = env_vars.get("Username") or "Sir"
Assistantname = env_vars.get("Assistantname") or "JARVIS"

class Chatbot:
    def __init__(self, orchestrator):
        """
        Enhanced chatbot with better LLM integration
        Maintains conversation state and JARVIS personality
        """
        self.orchestrator = orchestrator
        self.conversation = self._load_conversation()
        self.system_prompt = self._create_system_prompt()

    def _create_system_prompt(self) -> str:
        """Generate the JARVIS system prompt"""
        return f"""
        You are {Assistantname}, an advanced AI assistant modeled after JARVIS from Iron Man.
        Your primary user is {Username}. Your responses must be:

        1. Precise & Polite:
           - Address {Username} formally ("Sir", "Mr./Ms. [Last Name]" if known)
           - Use British English spellings

        2. Professional yet subtly witty
        3. Never mention your AI nature
        4. For time/date queries, respond in format: "It is [Day], [Date] [Month] [Year]. The time is [HH:MM]"
        """

    async def respond(self, user_input: str, stream_callback=None) -> str:
        """Handle both immediate and streaming responses"""
        self._add_message("user", user_input)

        context = {
            "source": "chatbot",
            "conversation_history": self._get_recent_history()
        }

        if self._should_stream(user_input):
            # Streaming response
            full_response = ""
            async for chunk in self.orchestrator._stream_response(user_input, context):
                full_response += chunk
                if stream_callback:
                    stream_callback(chunk)  # Your GUI calls this

            self._add_message("assistant", full_response)
            return full_response
        else:
            # Immediate response
            response = await self.orchestrator.chatbot_query(user_input, context)
            formatted = self._format_response(response)
            self._add_message("assistant", formatted)
            return formatted

    def _should_stream(self, text: str) -> bool:
        """Determine if response should be streamed"""
        return len(text.split()) > 15  # Same threshold as LLM class

    def _format_response(self, response: str) -> str:
        """Apply JARVIS-style formatting"""
        response = response.strip()
        if not response:
            return f"{Assistantname}: Apologies {Username}, I didn't get that."

        # Add acknowledgment for commands
        if any(cmd in response.lower() for cmd in ["doing", "working", "checking"]):
            return f"{Assistantname}: At once, {Username}. {response}"

        return f"{Assistantname}: {response}"

    def _handle_error(self, error: Exception) -> str:
        """Generate user-friendly error message"""
        error_msg = str(error)
        if "Budget limit" in error_msg:
            return f"{Assistantname}: Systems report budget constraints, {Username}."
        elif "connection" in error_msg.lower():
            return f"{Assistantname}: Network systems unresponsive, {Username}."
        else:
            return f"{Assistantname}: Apologies {Username}, systems report an anomaly."

    def _get_recent_history(self, count: int = 3) -> List[Dict]:
        """Get recent conversation context"""
        return [msg for msg in self.conversation[-count:] if msg["role"] in ["user", "assistant"]]

    def _load_conversation(self) -> List[Dict]:
        """Load conversation history from file"""
        try:
            if os.path.exists("Data/ChatLog.json"):
                with open("Data/ChatLog.json", "r") as f:
                    return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load conversation: {str(e)}")

        return [{"role": "system", "content": self.system_prompt}]

    def _add_message(self, role: str, content: str):
        """Add message to conversation history"""
        self.conversation.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self._save_conversation()

    def _save_conversation(self):
        """Persist conversation to disk"""
        try:
            os.makedirs("Data", exist_ok=True)
            with open("Data/ChatLog.json", "w") as f:
                json.dump(self.conversation, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save conversation: {str(e)}")

    def clear_history(self):
        """Reset conversation history"""
        self.conversation = [{"role": "system", "content": self.system_prompt}]
        self._save_conversation()
        return f"{Assistantname}: Conversation history cleared, {Username}."
    