import asyncio
import datetime
import email
from email.header import decode_header
from email.mime.text import MIMEText
import html
import imaplib
import json
import logging
import re
import random
import smtplib
from typing import Tuple
import webbrowser
from bs4 import BeautifulSoup
from groq import Groq
import groq
import pyautogui
import requests
from json import load, dump
from dotenv import dotenv_values
from googlesearch import search
import openai
import imaplib
import smtplib
import email
import logging
import re
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from email.header import decode_header
from typing import Tuple, List, Optional, Dict
from Backend.LLM import LLMOrchestrator  # Using OpenAI's ChatCompletion API
import re
import logging
from typing import Optional, AsyncGenerator
from dataclasses import asdict


# Load environment variables
env_vars = dotenv_values(".env")
Username = env_vars.get("Username") or "Sir"  # Default JARVIS-style address
Assistantname = env_vars.get("Assitantname") or "JARVIS"
GOOGLE_MAPS_API_KEY = env_vars.get("GOOGLE_MAPS_API_KEY")
WEATHER_API_KEY = env_vars.get("WEATHER_API_KEY")
NEWS_API_KEY = env_vars.get("NEWS_API_KEY")


EMAIL_USER = env_vars.get("EMAIL_USER")
EMAIL_PASS = env_vars.get("EMAIL_PASS")

# Global message log
try:
    with open(r"Data\ChatLog.json", "r") as f:
        messages = load(f)
except FileNotFoundError:
    with open(r"Data\ChatLog.json", "w") as f:
        dump([], f)
    messages = []

# Initialize assistant memory
assistant_memory = {}
    
try:
    with open("Data/contact_map.json", "r") as f:
        contact_map = json.load(f)
except FileNotFoundError:
    contact_map = {}

def get_time_of_day():
    """Return appropriate greeting based on time of day."""
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    return "evening"

# JARVIS-style system prompt
System = f"""You are {Assistantname}, an advanced AI assistant modeled after JARVIS from Iron Man. 
Your primary user is {Username}. Your responses should be:

1. Polite yet concise - address {Username} properly ("Sir", "Mr./Ms. [Last Name]")
2. Technically precise with occasional dry wit when appropriate
3. Never verbose unless explicitly asked for details
4. Professional yet personable
5. Use British English spellings and sophisticated phrasing

*** Key Directives ***
- Only provide time/date when explicitly asked
- Reply exclusively in English
- Never mention your training data or AI nature
- For real-time data, ensure absolute accuracy
- Present information cleanly without extra notes
- Be proactive but not overbearing
"""

# --- SystemChatBot initialization ---
SystemChatBot = [
    {"role": "system", "content": System},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": f"Good {get_time_of_day()}, {Username}. How may I assist you today?"}
]

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

def Information():
    """Return current time information."""
    now = datetime.datetime.now()
    data = (
        f"Use This Real-time Information if needed,\n"
        f"Day: {now.strftime('%A')}\n"
        f"Date: {now.strftime('%d')}\n"
        f"Month: {now.strftime('%B')}\n"
        f"Year: {now.strftime('%Y')}\n"
        f"Time: {now.strftime('%H')} hours : {now.strftime('%M')} minutes : {now.strftime('%S')} seconds.\n"
    )
    return data




class EmailHandler:
    def __init__(self, orchestrator: LLMOrchestrator):
        self.orchestrator = orchestrator
        self.pending_drafts = {}  # Stores unconfirmed emails by user_id
        self.IMPORTANT_KEYWORDS = ["urgent", "important", "action required", "meeting", "deadline"]

    async def fetch_emails(self, email_user: str, email_pass: str, limit: int = 5) -> List[str]:
        """Fetch and prioritize important emails with better error handling"""
        try:
            with imaplib.IMAP4_SSL("imap.gmail.com") as imap:
                imap.login(email_user, email_pass)
                imap.select("inbox")

                since_date = (datetime.now() - timedelta(days=2)).strftime("%d-%b-%Y")
                status, messages = imap.search(None, f'(SINCE "{since_date}")')

                if status != "OK":
                    return ["Unable to fetch emails at the moment."]

                important_msgs = []
                for msg_id in reversed(messages[0].split()[-limit:]):
                    _, msg_data = imap.fetch(msg_id, "(RFC822)")
                    if msg := self._process_email(msg_data):
                        important_msgs.append(msg)

                return important_msgs or ["No important emails in the last 48 hours."]

        except Exception as e:
            logging.error(f"Email fetch error: {str(e)}", exc_info=True)
            return [f"Error checking email: {str(e)}"]

    def _process_email(self, msg_data) -> Optional[str]:
        """Extract and format important email content"""
        for part in msg_data:
            if isinstance(part, tuple):
                msg = email.message_from_bytes(part[1])
                subject, encoding = decode_header(msg["Subject"])[0]
                subject = subject.decode(encoding or "utf-8") if isinstance(subject, bytes) else subject

                from_ = msg.get("From")
                body = self._extract_email_body(msg)

                if self._is_important(subject, body):
                    preview = body[:150].replace('\n', ' ').replace('\r', ' ')
                    return (f"🔔 From: {from_}\n"
                            f"📌 Subject: {subject}\n"
                            f"📝 Preview: {preview}...\n")
        return None

    def _extract_email_body(self, msg) -> str:
        """Extract text content from email"""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    return part.get_payload(decode=True).decode(errors="ignore")
        return msg.get_payload(decode=True).decode(errors="ignore")

    def _is_important(self, subject: str, body: str) -> bool:
        """Check if email contains important keywords"""
        content = f"{subject.lower()} {body.lower()}"
        return any(keyword in content for keyword in self.IMPORTANT_KEYWORDS)

    async def compose_email(self, recipient: str, user_prompt: str) -> Tuple[bool, str]:
        """Generate email draft using LLM"""
        try:
            llm_prompt = (
                f"Compose professional email to {recipient} based on:\n"
                f"USER REQUEST: {user_prompt}\n\n"
                "Format as:\nSubject: <clear subject line>\n\nBody: <well-structured email>"
            )

            response = await self.orchestrator.query_llm(
                prompt=llm_prompt,
                context={
                    "source": "email_composition",
                    "preferred_llm": "anthropic",  # Better for structured writing
                    "response_format": "email"
                }
            )

            if not response.get("success"):
                return False, "Failed to generate email draft."

            return self._parse_llm_response(response["response"], recipient)

        except Exception as e:
            logging.error(f"Email composition error: {str(e)}", exc_info=True)
            return False, f"Error generating email: {str(e)}"

    def _parse_llm_response(self, response: str, recipient: str) -> Tuple[bool, str]:
        """Extract subject and body from LLM response"""
        subject_match = re.search(r"Subject:\s*(.+)", response, re.IGNORECASE)
        body_match = re.search(r"Body:\s*(.+)", response, re.IGNORECASE | re.DOTALL)

        if not (subject_match and body_match):
            return False, "Couldn't parse email format from response."

        draft = {
            "to": recipient,
            "subject": subject_match.group(1).strip(),
            "body": body_match.group(1).strip()
        }

        # Store draft with expiration (1 hour)
        self.pending_drafts[recipient] = {
            "draft": draft,
            "expires": datetime.now() + timedelta(hours=1)
        }

        return True, self._format_draft_confirmation(draft)

    def _format_draft_confirmation(self, draft: Dict) -> str:
        """Format draft for user confirmation"""
        return (f"📧 Email Draft Ready:\n\n"
                f"To: {draft['to']}\n"
                f"Subject: {draft['subject']}\n\n"
                f"Body:\n{draft['body']}\n\n"
                f"Should I send this?")

    async def send_email(self, email_user: str, email_pass: str, recipient: str) -> Tuple[bool, str]:
        """Send a pending email draft"""
        if recipient not in self.pending_drafts:
            return False, "No pending email draft for this recipient."

        draft = self.pending_drafts[recipient]["draft"]
        
        try:
            msg = MIMEText(draft["body"])
            msg['Subject'] = draft["subject"]
            msg['From'] = email_user
            msg['To'] = recipient

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(email_user, email_pass)
                server.send_message(msg)

            del self.pending_drafts[recipient]  # Clear after sending
            return True, "Email sent successfully."

        except Exception as e:
            logging.error(f"Email send error: {str(e)}", exc_info=True)
            return False, f"Failed to send email: {str(e)}"

    def clear_draft(self, recipient: str) -> bool:
        """Clear a pending draft"""
        if recipient in self.pending_drafts:
            del self.pending_drafts[recipient]
            return True
        return False


def GoogleSearch(query):
    """Enhanced Google search with error handling."""
    try:
        results = list(search(query, advanced=True, num_results=3))
        if not results:
            return None
        answer = "Search Results:\n"
        for i, result in enumerate(results, 1):
            answer += f"{i}. {result.title}\n   {result.description}\n   {result.url}\n\n"
        return answer
    except Exception as e:
        print(f"Search error: {e}")
        return None

def get_weather(location):
    """Fetch weather data for a given location using OpenWeatherMap API."""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={WEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            weather = data['weather'][0]['description']
            temp = data['main']['temp']
            return f"The weather in {location} is {weather} with a temperature of {temp}°C."
        else:
            return f"Failed to retrieve weather information. Status code: {response.status_code}"
    except Exception as e:
        return f"Error fetching weather data: {e}"

def get_news(country_code="in"):
    """Fetch the latest news headlines for the given country using NewsAPI."""
    url = f"https://newsapi.org/v2/top-headlines?country={country_code}&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            headlines = [article['title'] for article in data['articles']]
            return "Here are the latest news headlines:\n" + "\n".join(headlines)
        else:
            return f"Failed to retrieve news headlines. Status code: {response.status_code}"
    except Exception as e:
        return f"Error fetching news data: {e}"

def extract_location(prompt):
    """Extract location from a weather-related query."""
    pattern = r"(?:weather|forecast)\s+(?:in|for|at)?\s*([\w\s]+)"
    match = re.search(pattern, prompt, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def analyze_stock(stock_symbol):
    """Enhanced stock analysis with Indian market support."""
    ALPHA_VANTAGE_KEY = env_vars.get("ALPHA_VANTAGE_KEY")
    if not ALPHA_VANTAGE_KEY:
        return "Stock API key not configured in .env file"
    if not any(ext in stock_symbol.upper() for ext in ['.NS', '.BO']):
        stock_symbol += '.NS'
    base_url = "https://www.alphavantage.co/query"
    try:
        quote_params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': stock_symbol,
            'apikey': ALPHA_VANTAGE_KEY
        }
        quote_data = requests.get(base_url, params=quote_params).json()
        if 'Global Quote' not in quote_data:
            return f"No data found for {stock_symbol}. Try: {stock_symbol.replace('.NS','.BO')} for BSE"
        quote = quote_data['Global Quote']
        current_price = float(quote['05. price'])
        fund_params = {
            'function': 'OVERVIEW',
            'symbol': stock_symbol,
            'apikey': ALPHA_VANTAGE_KEY
        }
        fund_data = requests.get(base_url, params=fund_params).json()
        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30)))  # IST
        market_open = (
            now.weekday() < 5 and
            datetime.time(9, 15) <= now.time() <= datetime.time(15, 30)
        )
        if not market_open:
            return (
                f"{get_jarvis_phrase('acknowledge')} Indian markets are currently closed (NSE hours: 9:15 AM - 3:30 PM IST).\n"
                f"Last price for {stock_symbol}: ₹{quote['05. price']}"
            )
        if not fund_data:
            return f"Could not analyze {stock_symbol} fundamentals."
        analysis = f"{get_jarvis_phrase('acknowledge')} Analysis for {stock_symbol}:\n"
        analysis += f"- Current Price: ₹{current_price:,.2f}\n"
        analysis += f"- 52 Week Range: ₹{float(fund_data.get('52WeekLow', 0)):,.2f} - ₹{float(fund_data.get('52WeekHigh', 0)):,.2f}\n"
        analysis += f"- Market Cap: ₹{format_market_cap(fund_data.get('MarketCapitalization'))}\n"
        analysis += f"- P/E Ratio: {fund_data.get('PERatio', 'N/A')}\n"
        analysis += f"- Dividend Yield: {fund_data.get('DividendYield', '0')}%\n"
        analysis += f"- Sector: {fund_data.get('Sector', 'N/A')}\n"
        suggestion = "\n\nInvestment Perspective: "
        try:
            pe_ratio = float(fund_data['PERatio'])
            dividend_yield = float(fund_data['DividendYield'])
            fifty_two_high = float(fund_data['52WeekHigh'])
            sector = fund_data.get('Sector', '').lower()
            if pe_ratio < 18 and dividend_yield > 1.5:
                suggestion += (
                    f"{stock_symbol.replace('.NS','')} appears reasonably valued for Indian markets "
                    f"(P/E: {pe_ratio:.1f} vs sector avg ~22). The {dividend_yield}% yield is attractive for the {sector} sector, Sir."
                )
            elif current_price < fifty_two_high * 0.85:
                suggestion += (
                    f"Currently {100*(1-current_price/fifty_two_high):.1f}% below 52-week high. "
                    f"Could be an accumulation opportunity if the {sector} sector outlook is positive."
                )
            else:
                suggestion += (
                    f"Valuation appears full at current levels. "
                    f"Consider waiting for a better entry point in this {sector} stock, Sir."
                )
        except (TypeError, ValueError):
            suggestion += "Incomplete fundamental data. Refer to NSE/BSE filings for the full picture, Sir."
        disclaimer = (
            "\n\nNote: Analysis based on Indian market parameters. Sector P/E averages from NIFTY indices. "
            "Not a recommendation - consult your SEBI-registered advisor."
        )
        return analysis + suggestion + disclaimer
    except Exception as e:
        return (
            f"{get_jarvis_phrase('error')} Analysis failed. Try direct links:\n"
            f"NSE: https://www.nseindia.com/get-quotes/equity?symbol={stock_symbol.replace('.NS','')}\n"
            f"BSE: https://www.bseindia.com/stock-share-price/{stock_symbol.replace('.BO','')}/"
        )

def format_market_cap(value):
    """Format market cap in Indian style (Cr, Lacs)"""
    try:
        value = float(value)
        if value >= 10000000:
            return f"{value/10000000:,.2f} Cr"
        elif value >= 100000:
            return f"{value/100000:,.2f} Lacs"
        return f"{value:,.2f}"
    except:
        return "N/A"


def clean_html_instruction(instr: str) -> str:
    return BeautifulSoup(html.unescape(instr), "html.parser").get_text()

def suggest_mode_by_distance(distance_km: float) -> str:
    """Basic mode suggestion logic"""
    if distance_km < 1:
        return "walking"
    elif distance_km < 5:
        return "bicycling"
    elif distance_km < 25:
        return "driving"
    return "transit"

async def get_route_with_llm_analysis(
    origin: str = None,
    destination: str = None,
    mode: str = None,
    control_center: LLMOrchestrator = None
) -> str:
    """
    Enhanced route finder with LLM-powered analysis of all available options
    Returns optimized route recommendation with detailed analysis
    """
    base_url = "https://maps.googleapis.com/maps/api/directions/json"
    valid_modes = ["driving", "walking", "bicycling", "transit"]
    
    # Handle memory fallbacks
    if not origin:
        origin = assistant_memory.get("last_origin")
    if not destination:
        destination = assistant_memory.get("last_destination")
    
    if not origin or not destination:
        return "🧭 Please specify both origin and destination."

    try:
        # First get basic route info to determine distance
        params = {
            "origin": origin,
            "destination": destination,
            "key": GOOGLE_MAPS_API_KEY,
            "mode": "driving"  # Default for initial check
        }
        
        res = requests.get(base_url, params=params)
        data = res.json()
        
        if data["status"] != "OK":
            return f"❌ Route not found: {data.get('status', 'Unknown error')}"
        
        route = data["routes"][0]["legs"][0]
        distance_km = float(re.findall(r"[\d.]+", route["distance"]["text"])[0])
        
        # If no mode specified, get ALL possible routes
        if not mode or mode.lower() not in valid_modes:
            all_routes = []
            
            for transport_mode in valid_modes:
                params["mode"] = transport_mode
                res = requests.get(base_url, params=params)
                data = res.json()
                
                if data["status"] == "OK":
                    route_data = data["routes"][0]["legs"][0]
                    all_routes.append({
                        "mode": transport_mode,
                        "distance": route_data["distance"]["text"],
                        "duration": route_data["duration"]["text"],
                        "steps": [
                            clean_html_instruction(step["html_instructions"]) 
                            for step in route_data["steps"]
                        ],
                        "traffic_info": route_data.get("duration_in_traffic", {}).get("text", "N/A")
                    })
            
            # Get LLM to analyze and recommend best option
            if all_routes and control_center:
                analysis_prompt = (
                    f"Analyze these route options from {origin} to {destination}:\n"
                    f"{json.dumps(all_routes, indent=2)}\n\n"
                    "Consider: time, convenience, weather (if known), and user preferences. "
                    "Recommend the best option with a brief rationale."
                )
                
                llm_response = await control_center.query_llm(
                    prompt=analysis_prompt,
                    context={
                        "source": "route_analysis",
                        "preferred_llm": "claude",  # Better for analysis
                        "response_format": "natural"
                    }
                )
                
                if llm_response.get("success"):
                    best_mode = next(
                        (r["mode"] for r in all_routes 
                         if r["mode"] in llm_response["response"].lower()),
                        suggest_mode_by_distance(distance_km)
                    )
                    
                    # Format final response with LLM analysis
                    selected_route = next(r for r in all_routes if r["mode"] == best_mode)
                    return format_route_response(selected_route, llm_response["response"])
            
            # Fallback if LLM analysis fails
            best_mode = suggest_mode_by_distance(distance_km)
            selected_route = next(r for r in all_routes if r["mode"] == best_mode)
            return format_route_response(selected_route)
        
        # Handle specific mode request
        if mode.lower() not in valid_modes:
            return f"❌ Invalid mode '{mode}'. Supported: {', '.join(valid_modes)}"
        
        params["mode"] = mode
        res = requests.get(base_url, params=params)
        data = res.json()
        
        if data["status"] != "OK":
            return f"❌ Couldn't get {mode} route: {data['status']}"
        
        route_data = data["routes"][0]["legs"][0]
        return format_route_response({
            "mode": mode,
            "distance": route_data["distance"]["text"],
            "duration": route_data["duration"]["text"],
            "steps": [
                clean_html_instruction(step["html_instructions"]) 
                for step in route_data["steps"]
            ],
            "traffic_info": route_data.get("duration_in_traffic", {}).get("text", "N/A")
        })
        
    
    except Exception as e:
        logging.error(f"Route error: {str(e)}")
    return f"🚨 Route planning failed: {str(e)}"

def format_route_response(route_data: dict, analysis: str = None) -> str:
    """Format route information into user-friendly response"""
    base_response = (
        f"📍 Route from {assistant_memory.get('last_origin')} to {assistant_memory.get('last_destination')}\n"
        f"🛣️ Transport: {route_data['mode'].title()}\n"
        f"📏 Distance: {route_data['distance']}\n"
        f"⏱️ Duration: {route_data['duration']}\n"
        f"🚦 Traffic: {route_data['traffic_info']}\n\n"
        f"🧭 Directions:\n" +
        "\n".join(f"{idx+1}. {step}" for idx, step in enumerate(route_data['steps'][:5]))  # Show first 5 steps
    )
    
    if analysis:
        return f"{base_response}\n\n🔍 Analysis:\n{analysis}"
    return base_response

# --- Real-time Search Engine (Updated) ---


class RealtimeEngine:
    def __init__(self, orchestrator: LLMOrchestrator):
        self.orchestrator = orchestrator
        self.query_handlers = {
            'weather': self._handle_weather,
            'news': self._handle_news,
            'route': self._handle_route,
            'stock': self._handle_stock,
            'email': self._handle_email
        }

    async def handle_query(self, prompt: str, stream_callback=None) -> str:
        """Main entry point for real-time queries"""
        processed_query = self._preprocess_query(prompt)
        
        try:
            # Route to specialized handler if available
            handler = self._identify_handler(processed_query)
            if handler:
                return await handler(processed_query)
            
            # Fallback to general search
            return await self._handle_general_search(processed_query, stream_callback)
            
        except Exception as e:
            error_msg = self._handle_error(e)
            logging.error(f"Realtime error: {str(e)}")
            return error_msg

    async def _handle_general_search(self, query: str, stream_callback=None) -> str:
        """Handle general search queries with LLM"""
        context = {
            "source": "realtime_search",
            "requires_fresh_data": True
        }

        if self._should_stream(query) and stream_callback:
            return await self._stream_response(query, context, stream_callback)
        
        response = await self.orchestrator.realtime_search(query, context)
        return self._format_response(response)

    async def _stream_response(self, query: str, context: dict, callback) -> str:
        """Handle streaming responses"""
        full_response = ""
        async for chunk in self.orchestrator._stream_response(query, context):
            full_response += chunk
            if callback:
                callback(chunk)
        return full_response.strip()

    def _identify_handler(self, query: str) -> Optional[callable]:
        """Identify specialized handler for query"""
        query_lower = query.lower()
        for key, handler in self.query_handlers.items():
            if any(trigger in query_lower for trigger in self._get_triggers(key)):
                return handler
        return None

    def _get_triggers(self, handler_type: str) -> list:
        """Get trigger words for each handler type"""
        return {
            'weather': ['weather', 'forecast', 'temperature'],
            'news': ['news', 'headlines'],
            'route': ['route', 'directions', 'how to reach'],
            'stock': ['stock', 'invest', 'share price'],
            'email': ['email', 'mail']
        }.get(handler_type, [])

    async def _handle_weather(self, query: str) -> str:
        """Specialized weather handler"""
        location = extract_location(query) or "New Delhi"
        weather = get_weather(location)
        return f"{get_jarvis_phrase('acknowledge')} {weather}"

    async def _handle_route(self, query: str) -> str:
        """Specialized route handler"""
        match = re.search(r"from\s+(.*?)\s+to\s+(.*)", query, re.IGNORECASE)
        if not match:
            return f"{get_jarvis_phrase('error')} Please specify origin and destination"
        
        origin, destination = match.group(1).strip(), match.group(2).strip()
        route = get_route_with_llm_analysis(origin, destination, self.orchestrator)
        
        return f"{get_jarvis_phrase('acknowledge')} {route}"

    def _preprocess_query(self, query: str) -> str:
        """Clean and normalize the query"""
        query = re.sub(rf"^{Assistantname}[,:]\s*", "", query, flags=re.IGNORECASE)
        return query.strip()

    def _should_stream(self, query: str) -> bool:
        """Determine if response should be streamed"""
        return len(query.split()) > 15

    def _format_response(self, response: str) -> str:
        """Apply JARVIS-style formatting"""
        if not response:
            return f"{get_jarvis_phrase('error')} No response received"
        
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        if len(lines) > 1:
            lines[0] = f"{get_jarvis_phrase('acknowledge')} {lines[0]}"
        return "\n".join(lines)

    def _handle_error(self, error: Exception) -> str:
        """Generate user-friendly error message"""
        error_msg = str(error)
        if "API" in error_msg:
            return f"{get_jarvis_phrase('error')} Service unavailable"
        elif "timeout" in error_msg.lower():
            return f"{get_jarvis_phrase('error')} Request timed out"
        return f"{get_jarvis_phrase('error')} Search failed"