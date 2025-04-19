import logging
from multiprocessing import context
import time
import os
import asyncio
from datetime import datetime
from typing import AsyncGenerator, Dict, List, Optional, Tuple
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
import numpy as np
from functools import lru_cache


class LLMOrchestrator:
    def __init__(self):
        # Initialize all LLM clients
        self.clients = {
            "openai": OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
            "anthropic": Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")),
        }
        
        # Cost configuration (in INR per 1K tokens)
        self.MODEL_COSTS = {
            "openai": {
                "gpt-4-turbo": 0.06,
                "gpt-3.5-turbo": 0.002,
            },
            "anthropic": {
                "claude-3-opus": 0.04,
                "claude-3-sonnet": 0.015,
            }
        }
        
        # Budget tracking
        self.total_spent = 0.0
        self.MONTHLY_BUDGET = 1000.0  # INR
        self.usage_history = []
        self.response_cache = {}
        
        # Personality profile
        self.personality_profile = {
            "tone": "professional",
            "verbosity": "normal",
            "humor": "none"
        }
        
        # Performance metrics
        self.performance_metrics = {
            "response_times": [],
            "error_rates": [],
            "user_satisfaction": []
        }
        
        # LLM capabilities
        self.llm_specializations = {
            "openai": {
                "strengths": ["creative", "general", "coding"],
                "default_model": "gpt-4-turbo",
                "fallback_model": "gpt-3.5-turbo"
            },
            "anthropic": {
                "strengths": ["reasoning", "safety", "long-form"],
                "default_model": "claude-3-opus",
                "fallback_model": "claude-3-sonnet"
            }
        }

    async def warmup_connections(self):
        """Pre-establish connections to LLM services"""
        warmup_prompts = {
            "openai": "Hello",
            "anthropic": "Greetings"
        }
        
        for llm, prompt in warmup_prompts.items():
            try:
                await self.query_llm(prompt, {"preferred_llm": llm})
            except:
                pass  # Silent fail for warmup

    async def query_llm(self, prompt: str, context: dict = None) -> dict:
        """Enhanced query with caching and budget awareness"""
        context = context or {}
        cache_key = self._generate_cache_key(prompt, context)
        
        # Check cache first
        if cache_key in self.response_cache:
            cached = self.response_cache[cache_key]
            if time.time() - cached["timestamp"] < 3600:  # 1 hour cache
                return cached["response"]
        
        # Check budget
        if self._is_budget_exhausted():
            return {
                "success": False,
                "error": f"Monthly budget of ₹{self.MONTHLY_BUDGET} exhausted",
                "budget_remaining": 0
            }
        
        # Select model
        selected_llm, selected_model = self._select_llm_model(prompt, context)
        
        try:
            start_time = time.time()
            response = await self._execute_query(selected_llm, selected_model, prompt)
            
            # Calculate cost
            tokens_used = self._estimate_tokens(response, selected_llm)
            cost = self._calculate_cost(selected_llm, selected_model, tokens_used)
            
            # Update budget
            self._update_budget(cost, selected_llm, selected_model, tokens_used)
            
            result = {
                "success": True,
                "llm": selected_llm,
                "model": selected_model,
                "response": response,
                "metadata": {
                    "tokens_used": tokens_used,
                    "cost": cost,
                    "budget_remaining": self.MONTHLY_BUDGET - self.total_spent,
                    "latency": time.time() - start_time
                }
            }
            
            # Cache successful responses
            self.response_cache[cache_key] = {
                "response": result,
                "timestamp": time.time()
            }
            
            return result
            
        except Exception as e:
            # Fallback logic
            if "fallback_model" in context.get("preferred_llm", ""):
                raise  # Already tried fallback
                
            fallback_model = self.llm_specializations[selected_llm]["fallback_model"]
            return await self.query_llm(prompt, {
                **context,
                "preferred_llm": f"{selected_llm}/{fallback_model}",
                "is_fallback": True
            })

    def _select_llm_model(self, prompt: str, context: dict) -> Tuple[str, str]:
        """Improved model selection with latency and budget awareness"""
        # Check for explicit preference
        if context.get("preferred_llm"):
            if "/" in context["preferred_llm"]:
                llm, model = context["preferred_llm"].split("/")
                return llm, model
            return context["preferred_llm"], self.llm_specializations[context["preferred_llm"]]["default_model"]
        
        # Analyze prompt needs
        task_type = self._analyze_task_type(prompt)
        remaining_budget = self.MONTHLY_BUDGET - self.total_spent
        
        # Latency scoring
        latency_scores = {
            "gpt-4-turbo": 2,  # Slower but powerful
            "gpt-3.5-turbo": 4, # Faster
            "claude-3-opus": 1,
            "claude-3-sonnet": 3
        }
        
        # Score available options
        options = []
        for llm, specs in self.llm_specializations.items():
            base_score = 3 if task_type in specs["strengths"] else 1
            model = specs["default_model"]
            cost_per_k = self.MODEL_COSTS[llm][model]
            
            # Budget-aware scoring
            budget_score = min(remaining_budget / (cost_per_k * 10), 3)
            # Add latency to score
            total_score = base_score + budget_score + latency_scores.get(model, 3)
            
            options.append({
                "llm": llm,
                "model": model,
                "score": total_score
            })
        
        # Prioritize lower cost when budget is low
        remaining_ratio = remaining_budget / self.MONTHLY_BUDGET
        if remaining_ratio < 0.2:  # When budget is low
            for option in options:
                cost_per_k = self.MODEL_COSTS[option["llm"]][option["model"]]
                option["score"] *= (1 + (1 - cost_per_k/max(cost_per_k, 0.01)))
        
        # Select best option
        best = max(options, key=lambda x: x["score"])
        return best["llm"], best["model"]

    async def _execute_query(self, llm: str, model: str, prompt: str):
        """Execute query with specific LLM"""
        if llm == "openai":
            return self.clients["openai"].chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            ).choices[0].message.content
            
        elif llm == "anthropic":
            return self.clients["anthropic"].messages.create(
                model=model,
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}]
            ).content[0].text

    async def chatbot_query(self, prompt: str, conversation_history: list = None) -> str:
        """Enhanced chatbot with personality and streaming"""
        context = {
            "source": "chatbot",
            "preferred_llm": "openai/gpt-4-turbo",
            "conversation_history": conversation_history or [],
            "task_type": self._analyze_task_type(prompt)
        }
        
        # Apply personality
        prompt = self._apply_personality(prompt)
        
        # Stream longer responses
        if self._should_stream_response(prompt):
            return await self._stream_response(prompt, context)
            
        response = await self.query_llm(prompt, context)
        if response["success"]:
            return response["response"]
        raise Exception(response.get("error", "Chatbot query failed"))

    async def realtime_search(self, query: str) -> str:
        """Enhanced search with caching"""
        context = {
            "source": "realtime_search",
            "preferred_llm": "anthropic/claude-3-sonnet",
            "task_type": "information_retrieval"
        }
        
        response = await self.query_llm(query, context)
        if response["success"]:
            return response["response"]
        raise Exception(response.get("error", "Realtime query failed"))

    async def _stream_response(self, prompt: str, context: dict) -> AsyncGenerator[str, None]:
        """Stream response chunks without early termination"""
        full_response = ""
        response_gen = await self._execute_query_streaming(
        context["preferred_llm"].split("/")[0],
        context["preferred_llm"].split("/")[1],
        prompt
    )
    
        async for chunk in response_gen:
            full_response += chunk
            yield chunk  # Yield each chunk as it arrives
    
    # Optional: Yield final assembled response if needed
        yield full_response.strip()

    def _apply_personality(self, prompt: str) -> str:
        """Modify prompt based on personality settings"""
        instructions = []
        
        if self.personality_profile["tone"] == "professional":
            instructions.append("Respond professionally")
        elif self.personality_profile["tone"] == "friendly":
            instructions.append("Respond warmly")
            
        if self.personality_profile["verbosity"] == "concise":
            instructions.append("Be extremely concise")
        elif self.personality_profile["verbosity"] == "detailed":
            instructions.append("Provide detailed explanations")
            
        if instructions:
            return f"{' '.join(instructions)}. {prompt}"
        return prompt

    def _should_stream_response(self, prompt: str) -> bool:
        """Determine if response should be streamed"""
        return len(prompt.split()) > 15

    def _generate_cache_key(self, prompt: str, context: dict) -> str:
        """Generate consistent cache key"""
        return f"{hash(prompt)}-{hash(frozenset(context.items())) if context else ''}"

    def _analyze_task_type(self, prompt: str) -> str:
        """Analyze the prompt to determine task type"""
        prompt = prompt.lower()
        
        if any(word in prompt for word in ["write", "create", "story", "poem"]):
            return "creative"
        elif any(word in prompt for word in ["code", "program", "algorithm"]):
            return "coding"
        elif any(word in prompt for word in ["math", "calculate", "equation"]):
            return "math"
        elif any(word in prompt for word in ["explain", "why", "reason"]):
            return "reasoning"
        else:
            return "general"

    def _estimate_tokens(self, response: str, llm: str) -> int:
        """Naive token estimation (1 token ~ 4 characters in English)"""
        if not response:
            return 0
        char_count = len(response)
        return int(char_count / 4)


    def _calculate_cost(self, llm: str, model: str, tokens: int) -> float:
        """Calculate cost in INR"""
        cost_per_k = self.MODEL_COSTS[llm][model]
        return (tokens / 1000) * cost_per_k

    def _update_budget(self, cost: float, llm: str, model: str, tokens_used: int):
        """Update internal budget tracker"""
        self.total_spent += cost
        self.usage_history.append({
            "timestamp": datetime.now().isoformat(),
            "llm": llm,
            "model": model,
            "tokens_used": tokens_used,
            "cost": cost
        })


    def _is_budget_exhausted(self) -> bool:
        """Check if the current usage exceeds the monthly budget"""
        return self.total_spent >= self.MONTHLY_BUDGET


    def get_usage_report(self) -> dict:
        """Generate usage summary"""
        return {
            "total_spent": self.total_spent,
            "remaining_budget": self.MONTHLY_BUDGET - self.total_spent,
            "llm_breakdown": {
                llm: sum(
                    entry["cost"] for entry in self.usage_history 
                    if entry["llm"] == llm
                )
                for llm in self.MODEL_COSTS.keys()
            }
        }

    def get_performance_report(self) -> dict:
        """Generate performance analytics"""
        return {
            "avg_response_time": np.mean(self.performance_metrics["response_times"]),
            "error_rate": len(self.performance_metrics["error_rates"]) / 
                         max(len(self.performance_metrics["response_times"]), 1),
            "recent_errors": self.performance_metrics["error_rates"][-5:]
        }