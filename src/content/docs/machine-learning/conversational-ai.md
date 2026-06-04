---
title: "Conversational AI: Building Chat Systems That Work"
description: "Design and implement production conversational AI systems — from simple chatbots to complex multi-turn dialog management with context handling and error recovery."
---

Conversational AI spans from simple question-answering to complex multi-turn dialog. This guide covers the architecture, components, and best practices for building systems that feel natural and work reliably.

## Conversation Architecture

```python
class ConversationalSystem:
    def __init__(self, llm, memory, tools, config):
        self.llm = llm                  # Core language model
        self.memory = memory            # Conversation history
        self.tools = tools              # Available functions
        self.config = config            # System configuration
    
    def process_message(self, user_message: str) -> str:
        # 1. Load conversation context
        context = self.load_context()
        
        # 2. Analyze user intent
        intent = self.classify_intent(user_message, context)
        
        # 3. Execute based on intent
        if intent == "question":
            response = self.answer_question(user_message, context)
        elif intent == "command":
            response = self.execute_command(user_message, context)
        elif intent == "chit_chat":
            response = self.generate_response(user_message, context)
        else:
            response = self.handle_fallback(user_message, context)
        
        # 4. Update memory
        self.update_memory(user_message, response)
        
        return response
```

## Multi-Turn Dialog Management

```python
class DialogManager:
    def __init__(self, llm, dialog_state):
        self.llm = llm
        self.state = dialog_state  # Tracks slots, intent, etc.
    
    def process_turn(self, user_utterance: str) -> str:
        # Update dialog state with new utterance
        self.state.history.append({
            "role": "user",
            "content": user_utterance
        })
        
        # Check if all required slots are filled
        if self.state.intent and self.state.all_slots_filled():
            return self.execute_intent()
        
        # If slots are missing, ask for clarification
        missing = self.state.missing_required_slots()
        if missing:
            return self.ask_for_slots(missing)
        
        # Classify intent from utterance
        intent = self.classify_intent(user_utterance)
        self.state.intent = intent
        
        # Get required slots for this intent
        required = self.get_required_slots(intent)
        
        # Ask for missing slots
        missing = [s for s in required if s not in self.state.slots]
        if missing:
            return self.ask_for_slots(missing)
        
        # Default response
        return self.generate_response()
    
    def execute_intent(self):
        """Execute the user's request."""
        intent = self.state.intent
        slots = self.state.slots
        
        response = self.call_service(intent, slots)
        
        self.state.history.append({
            "role": "assistant",
            "content": response
        })
        
        return response
    
    def reset(self):
        """Reset dialog state for new conversation."""
        self.state = DialogState()
```

## Context Management

```python
class ConversationMemory:
    def __init__(self, max_tokens=16000):
        self.max_tokens = max_tokens
        self.history = []  # List of (role, message) tuples
    
    def add(self, role: str, message: str):
        self.history.append((role, message))
    
    def get_context(self) -> List[dict]:
        """Get conversation history formatted for LLM."""
        return [
            {"role": role, "content": content}
            for role, content in self.history
        ]
    
    def prune(self, system_prompt: str = ""):
        """Prune history to fit within token limit."""
        current_tokens = count_tokens(system_prompt)
        
        # Count history tokens
        history_tokens = 0
        for role, content in self.history:
            history_tokens += count_tokens(content)
        
        # Prune from beginning if over limit
        while current_tokens + history_tokens > self.max_tokens and self.history:
            removed_role, removed_content = self.history.pop(0)
            history_tokens -= count_tokens(removed_content)
        
        return self.history
    
    def summarize_old_messages(self, summary_prompt: str):
        """Summarize old messages to preserve information density."""
        # Take first N messages, summarize, replace with summary
        old_messages = self.history[:10]
        summary = self.summarize(old_messages, summary_prompt)
        
        self.history = [{"role": "assistant", "content": f"[Summary: {summary}]"}] + self.history[10:]
```

## Intent Classification

```python
class IntentClassifier:
    def __init__(self, llm, intents):
        self.llm = llm
        self.intents = intents  # List of (intent_name, description)
    
    def classify(self, user_message: str) -> str:
        """Classify user message into intent categories."""
        prompt = f"""Classify this message into one of these categories:
{chr(10).join(f"- {name}: {desc}" for name, desc in self.intents)}

Message: "{user_message}"

Category:"""
        
        response = self.llm.generate(prompt, temperature=0.0)
        
        # Parse response
        for name, _ in self.intents:
            if name.lower() in response.lower():
                return name
        
        return "unknown"
```

## Slot Filling

```python
class SlotFiller:
    def __init__(self, slots):
        self.slots = slots  # {slot_name: extraction_prompt}
    
    def extract(self, user_message: str) -> dict:
        """Extract slot values from user message."""
        extracted = {}
        
        for slot_name, prompt in self.slots.items():
            full_prompt = f"""{prompt}

User message: "{user_message}"

Extracted {slot_name}:"""
            
            value = self.llm.generate(full_prompt, temperature=0.0)
            extracted[slot_name] = value.strip()
        
        return extracted
    
    def update_state(self, current_state: dict, new_extractions: dict) -> dict:
        """Merge new extractions with existing state."""
        updated = current_state.copy()
        
        for slot, value in new_extractions.items():
            if value and value != "None":
                updated[slot] = value
        
        return updated
```

## Error Handling and Recovery

```python
class RobustConversation:
    def __init__(self, llm, config):
        self.llm = llm
        self.config = config
        self.max_retries = 3
    
    def safe_generate(self, prompt: str) -> str:
        """Generate with error handling and retries."""
        for attempt in range(self.max_retries):
            try:
                return self.llm.generate(prompt, temperature=0.7)
            except RateLimitError:
                wait_time = (attempt + 1) * 2
                time.sleep(wait_time)
            except TimeoutError:
                if attempt == self.max_retries - 1:
                    return self.fallback_response()
                time.sleep(1)
            except Exception as e:
                log_error(e)
                return self.error_response()
        
        return self.fallback_response()
    
    def handle_misunderstanding(self, user_message: str, context: dict) -> str:
        """Handle when user message doesn't make sense."""
        retry_prompt = f"""The user's message seems unclear given the context.
        
Context: {context}
User message: "{user_message}"

Generate a polite clarification request:"""
        
        return self.llm.generate(retry_prompt)
    
    def fallback_response(self) -> str:
        """Generic fallback for failures."""
        fallbacks = [
            "I'm having trouble understanding. Could you rephrase that?",
            "Let me get back to you on that. What else can I help with?",
            "I want to make sure I understand correctly. Could you provide more details?",
        ]
        return random.choice(fallbacks)
```

## Context Window Optimization

```python
class OptimizedContext:
    def __init__(self, llm, max_tokens=16000):
        self.llm = llm
        self.max_tokens = max_tokens
    
    def build_prompt(
        self,
        system_prompt: str,
        conversation: List[dict],
        current_query: str,
        relevant_docs: List[str] = None
    ) -> str:
        """Build optimized prompt within token limit."""
        prompt_parts = []
        remaining = self.max_tokens
        
        # Reserve space for response
        remaining -= 512
        
        # Add system prompt first (always included)
        prompt_parts.append(f"System: {system_prompt}")
        remaining -= count_tokens(system_prompt)
        
        # Add relevant documents if provided
        if relevant_docs:
            for doc in relevant_docs:
                if count_tokens(doc) < remaining - 500:
                    prompt_parts.append(f"Context: {doc}")
                    remaining -= count_tokens(doc)
        
        # Add conversation history (most recent first)
        for msg in reversed(conversation[-20:]):  # Last 20 messages
            msg_text = f"{msg['role'].title()}: {msg['content']}"
            msg_tokens = count_tokens(msg_text)
            
            if msg_tokens < remaining - 200:
                prompt_parts.append(msg_text)
                remaining -= msg_tokens
        
        # Add current query
        prompt_parts.append(f"Current question: {current_query}")
        
        return "\n\n".join(prompt_parts)
```

## Evaluation Metrics

```python
class ConversationEvaluator:
    def evaluate_conversation(self, conversation: List[dict], references: dict) -> dict:
        """Evaluate a conversation on multiple dimensions."""
        metrics = {}
        
        # Response relevance
        metrics["relevance"] = self.evaluate_relevance(conversation)
        
        # Context coherence
        metrics["coherence"] = self.evaluate_coherence(conversation)
        
        # Task completion (if applicable)
        if "task_completion" in references:
            metrics["task_completion"] = self.evaluate_task_completion(
                conversation, references["task_completion"]
            )
        
        # Engagement (length, follow-up questions)
        metrics["engagement"] = self.evaluate_engagement(conversation)
        
        # Safety (harmful content detection)
        metrics["safety"] = self.evaluate_safety(conversation)
        
        return metrics
    
    def evaluate_relevance(self, conversation: List[dict]) -> float:
        """Score how relevant responses are to user queries."""
        scores = []
        
        for msg in conversation:
            if msg["role"] == "assistant":
                # LLM-based evaluation
                score = self.llm_evaluate(
                    "How relevant is this response to the user's query?",
                    msg["content"]
                )
                scores.append(score)
        
        return mean(scores)
```

## Production Considerations

### Rate Limiting

```python
class RateLimiter:
    def __init__(self, requests_per_minute=60, tokens_per_minute=100000):
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        self.requests = deque()
        self.tokens = deque()
    
    def check(self, num_tokens: int) -> bool:
        """Check if request is within limits."""
        now = time.time()
        
        # Remove old entries
        while self.requests and now - self.requests[0] > 60:
            self.requests.popleft()
        
        while self.tokens and now - self.tokens[0] > 60:
            self.tokens.popleft()
        
        # Check limits
        if len(self.requests) >= self.rpm:
            return False
        
        if sum(self.tokens) + num_tokens > self.tpm:
            return False
        
        return True
    
    def record(self, num_tokens: int):
        """Record a completed request."""
        now = time.time()
        self.requests.append(now)
        self.tokens.append((now, num_tokens))
```

### A/B Testing

```python
class ConversationABTest:
    def __init__(self, variant_a, variant_b, traffic_split=0.5):
        self.variant_a = variant_a  # Control
        self.variant_b = variant_b  # Test
        self.split = traffic_split
        self.results = {"a": [], "b": []}
    
    def route_request(self, user_id: str) -> str:
        """Route user to A or B variant."""
        hash_value = hash(user_id) % 100
        return "b" if hash_value < self.split * 100 else "a"
    
    def record_outcome(self, variant: str, user_id: str, metrics: dict):
        """Record outcome for analysis."""
        self.results[variant].append({
            "user_id": user_id,
            "metrics": metrics,
        })
    
    def compute_statistical_significance(self) -> dict:
        """Compute significance of differences."""
        # Statistical tests for A/B comparison
        pass
```

### Feedback Integration

```python
class FeedbackSystem:
    def __init__(self, storage):
        self.storage = storage  # Database for feedback
    
    def record_feedback(self, conversation_id: str, feedback: dict):
        """Record user feedback."""
        self.storage.insert("feedback", {
            "conversation_id": conversation_id,
            "feedback_type": feedback["type"],  # "thumbs_up", "thumbs_down", "correction"
            "rating": feedback.get("rating"),
            "comment": feedback.get("comment"),
            "timestamp": time.time(),
        })
    
    def analyze_feedback(self, days: int = 7) -> dict:
        """Analyze feedback patterns."""
        recent_feedback = self.storage.query(f"""
            SELECT * FROM feedback
            WHERE timestamp > {time.time() - days * 86400}
        """)
        
        return {
            "total_feedback": len(recent_feedback),
            "positive_rate": sum(f["rating"] for f in recent_feedback) / len(recent_feedback),
            "common_issues": self.identify_issues(recent_feedback),
        }
```

Building conversational AI requires balancing multiple concerns: understanding user intent, maintaining coherent dialog, handling errors gracefully, and continuously improving based on feedback. The patterns and techniques here provide a foundation for production-ready systems.