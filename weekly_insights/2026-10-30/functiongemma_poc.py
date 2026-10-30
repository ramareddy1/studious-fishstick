"""
Proof-of-Concept: On-Device Function Calling with Small Language Models
Based on: "Extending FunctionGemma for Practical On-Device Mobile Function Calling"
arXiv: 2609.25373 | EMNLP 2026

This script demonstrates the core mechanic of the paper:
- Define a set of device-control tools in a JSON schema
- Format a prompt using the FunctionGemma-style function calling template
- Parse the model's output as a structured function call

Run modes:
  1. Demo mode (default, no model needed): shows the prompt format + mock output
  2. Inference mode: loads google/functiongemma-270m-it from Hugging Face
     pip install transformers accelerate
     then set DEMO_MODE = False
"""

import json

DEMO_MODE = True  # set False to run actual model inference

# ---------------------------------------------------------------------------
# Tool definitions (mobile-action schema, as used in the paper)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "send_message",
        "description": "Send a text message to a contact.",
        "parameters": {
            "type": "object",
            "properties": {
                "contact": {"type": "string", "description": "Recipient name or phone number"},
                "message": {"type": "string", "description": "Message body"},
            },
            "required": ["contact", "message"],
        },
    },
    {
        "name": "set_alarm",
        "description": "Set an alarm for a specific time.",
        "parameters": {
            "type": "object",
            "properties": {
                "time": {"type": "string", "description": "Time in HH:MM 24-hour format"},
                "label": {"type": "string", "description": "Optional alarm label"},
            },
            "required": ["time"],
        },
    },
    {
        "name": "open_app",
        "description": "Open a mobile application by name.",
        "parameters": {
            "type": "object",
            "properties": {
                "app_name": {"type": "string", "description": "Name of the app to open"},
            },
            "required": ["app_name"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web for a query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    },
]


def build_functiongemma_prompt(tools: list, user_request: str) -> str:
    """
    Constructs a prompt in the FunctionGemma instruction format.
    The model expects tools serialized in the system turn, followed by the
    user request in the user turn.
    """
    tools_json = json.dumps(tools, indent=2)
    prompt = (
        "<bos><start_of_turn>system\n"
        "You are a helpful assistant with access to the following tools. "
        "Respond ONLY with a JSON object matching exactly one tool call.\n\n"
        f"Tools:\n{tools_json}<end_of_turn>\n"
        "<start_of_turn>user\n"
        f"{user_request}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )
    return prompt


def parse_function_call(raw_output: str) -> dict | None:
    """Extract and validate the function call JSON from model output."""
    try:
        # Strip markdown code fences if present
        text = raw_output.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        call = json.loads(text.strip())
        # Minimal validation: must have name + arguments
        if "name" in call and "arguments" in call:
            return call
    except json.JSONDecodeError:
        pass
    return None


def run_demo(user_request: str):
    """Run in demo mode: show the formatted prompt and a mocked model response."""
    prompt = build_functiongemma_prompt(TOOLS, user_request)

    print("=" * 60)
    print("USER REQUEST:", user_request)
    print("-" * 60)
    print("FORMATTED PROMPT (truncated for display):")
    # Show only the last part of the prompt to keep output readable
    lines = prompt.split("\n")
    for line in lines[-12:]:
        print(line)
    print("-" * 60)

    # Simulated output that a fine-tuned FunctionGemma 270M would produce
    mock_outputs = {
        "Send a message to Alice saying I'll be late": json.dumps(
            {"name": "send_message", "arguments": {"contact": "Alice", "message": "I'll be late"}},
            indent=2,
        ),
        "Set an alarm for 7:30 tomorrow": json.dumps(
            {"name": "set_alarm", "arguments": {"time": "07:30", "label": "Wake up"}},
            indent=2,
        ),
        "Open Spotify": json.dumps(
            {"name": "open_app", "arguments": {"app_name": "Spotify"}},
            indent=2,
        ),
    }
    mock_output = mock_outputs.get(
        user_request,
        json.dumps({"name": "web_search", "arguments": {"query": user_request}}, indent=2),
    )

    print("MODEL OUTPUT (mock):")
    print(mock_output)
    parsed = parse_function_call(mock_output)
    if parsed:
        print("\nPARSED CALL ->", parsed["name"], "with args:", parsed["arguments"])
    print("=" * 60)


def run_inference(user_request: str):
    """Load FunctionGemma 270M from HuggingFace and run actual inference."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    MODEL_ID = "google/functiongemma-270m-it"
    print(f"Loading {MODEL_ID} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,  # float32 for CPU safety; use bfloat16 on GPU
        device_map="auto",
    )

    prompt = build_functiongemma_prompt(TOOLS, user_request)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=1.0,
        )

    raw = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("USER REQUEST:", user_request)
    print("RAW MODEL OUTPUT:", raw)
    parsed = parse_function_call(raw)
    if parsed:
        print("PARSED CALL ->", parsed["name"], "with args:", parsed["arguments"])
    else:
        print("Could not parse a valid function call from output.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_requests = [
        "Send a message to Alice saying I'll be late",
        "Set an alarm for 7:30 tomorrow",
        "Open Spotify",
        "What is the weather in New York today?",
    ]

    if DEMO_MODE:
        print("Running in DEMO mode (no model download required).")
        print("Set DEMO_MODE = False in the script to run actual inference.\n")
        for req in test_requests:
            run_demo(req)
    else:
        for req in test_requests:
            run_inference(req)
            print()
