---
title: "Function Calling and Tool Use: Extending AI Capabilities"
description: "How AI models use function calls to interact with external tools and why structured outputs matter."
---

An LLM generates text. It can't execute code, query a database, or call an API directly. Function calling (also called tool use) bridges this gap: the model decides which function to call and with what parameters, then your system executes it.

## How Function Calling Works

```
User: "What's the weather in New York?"
     ↓
LLM sees available tools: [get_weather(city), search_web(query)]
     ↓
LLM outputs: {"tool": "get_weather", "city": "New York"}
     ↓
System calls: get_weather("New York")
Result: "75°F, Sunny"
     ↓
System passes result back to LLM
     ↓
LLM generates: "The weather in New York is 75°F and sunny."
```

## Defining Functions

Most APIs use JSON schema to describe available functions:

```json
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "Get the current weather for a location",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {
          "type": "string",
          "description": "The city name"
        },
        "units": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"],
          "description": "Temperature units"
        }
      },
      "required": ["city"]
    }
  }
}
```

The model understands what each function does and when to use it.

## Real-World Applications

### Multi-Step Tasks
```
User: "Book me a flight from NYC to LA next week"

Step 1: LLM calls search_flights(origin, destination, date_range)
Step 2: Reviews results, calls get_hotel_options(city, dates)
Step 3: Calls create_booking(flight_id, hotel_id)
```

### Data Retrieval
```
User: "How much did we spend on AWS last month?"

Step 1: LLM calls query_billing_system(service='AWS', month='2024-12')
Step 2: Receives $5,432.15
Step 3: Generates response with context
```

### Real-Time Information
```
User: "Who won the Super Bowl?"

Step 1: LLM calls search_web(query='Super Bowl winner 2024')
Step 2: Gets current information
Step 3: Provides accurate, up-to-date answer
```

## Function Calling vs. Prompt Engineering

**Naive Prompting:**
```
System: "If asked about weather, respond with reasonable guesses"
User: "What's the weather in New York?"
LLM: "It's probably around 70 degrees, maybe cloudy..."
```

**Function Calling:**
```
System: [defines get_weather function]
User: "What's the weather in New York?"
LLM: Calls get_weather("New York")
Result: Actual, accurate weather data
```

## Model Reliability

Not all models handle function calling equally:

| Model | Function Calling | Notes |
|-------|---|---|
| **GPT-4o** | Excellent | Reliable, follows schema strictly |
| **Claude 3.5 Sonnet** | Excellent | Very reliable, good parameter accuracy |
| **Gemini 1.5 Pro** | Good | Works well, occasional schema issues |
| **GPT-4o Mini** | Good | Reliable but less nuanced |
| **LLaMA 2** | Inconsistent | Requires specific fine-tuning |
| **Mistral** | Moderate | Follows schemas reasonably well |

## Common Patterns

### Parallel Function Calls
Some models can call multiple functions at once:

```
LLM output:
[
  {"tool": "get_price", "product": "widget"},
  {"tool": "get_inventory", "product": "widget"},
  {"tool": "get_reviews", "product": "widget"}
]

System executes all three in parallel, returns results
```

### Function Call Loops
```
User: "Find me a good hotel in Paris under $100/night"

Call 1: search_hotels(city="Paris", max_price=100)
Result: Found 5 hotels, but details are sparse

Call 2: get_hotel_details(hotel_id="paris_123")
Call 3: get_hotel_details(hotel_id="paris_456")
...

LLM: "Here are the best options with full details..."
```

### Error Handling
```
LLM calls: book_flight(flight_id=999)
System: "Error: Flight not found"
LLM: Calls search_flights(route, date) again with different params
```

## Best Practices

1. **Clear Descriptions:** Model quality depends on how well you describe functions
2. **Required Parameters:** Always mark truly required fields
3. **Enums for Validation:** Use enum for fixed options (units, status codes)
4. **Error Messages:** Return clear error messages when function calls fail
5. **Timeout Handling:** Set timeouts; don't let the model wait forever
6. **Rate Limiting:** Prevent models from making too many function calls
7. **Audit Trails:** Log which functions were called and with what parameters

## Implementation Example

```python
import anthropic

client = anthropic.Anthropic()

tools = [
    {
        "name": "calculator",
        "description": "Performs arithmetic operations",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["add", "subtract", "multiply"]},
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "required": ["operation", "a", "b"]
        }
    }
]

messages = [{"role": "user", "content": "What's 25 times 4?"}]

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    tools=tools,
    messages=messages
)

# Model might return: tool_use with operation="multiply", a=25, b=4
```

## Security Considerations

- **Validate Input:** Don't trust parameter values; validate them
- **Authorization:** Check if user can call that function
- **Rate Limiting:** Prevent API abuse
- **Audit Logging:** Track all function calls for compliance
- **Sandboxing:** Execute functions in isolated environment