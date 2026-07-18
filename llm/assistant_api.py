# llm/assistant_api.py

'''
Day 7 — LLM Assistant using Groq API (LLaMA 3.1 70B).
Production-quality version after migrating from Ollama + Mistral 7B.

Why we switched from Ollama to Groq:
  - Mistral 7B (local) struggled with multi-step reasoning
  - Confused units (said "minutes" instead of "seconds")
  - Couldn't reliably distinguish small talk from traffic questions
  - Called wrong tools for complex questions
  - LLaMA 3.1 70B via Groq handles all of these correctly
  - Groq is free tier, near-instant responses (~1-2 seconds)
  - Same tool definitions, same architecture — just a better model

This file is a drop-in replacement for assistant.py.
The ask_once() interface is identical so dashboard.py works with both.
'''

import os
import json
from groq import Groq
from llm.tools import TOOLS, dispatch_tool

from llm.rag import retrieve

# convert our tool definitions to OpenAI-compatible format
# Groq uses the same format as OpenAI
def convert_tools(tools):
    return [
        {
            'type'    : 'function',
            'function': {
                'name'       : t['name'],
                'description': t['description'],
                'parameters' : t['input_schema']
            }
        }
        for t in tools
    ]

GROQ_TOOLS = convert_tools(TOOLS)

SYSTEM_PROMPT = '''You are TrafficPulse Assistant, an AI traffic operations analyst
for a smart city traffic management system monitoring a 3-intersection urban network.

You have access to real data tools. Always use them — never guess or invent numbers.

NETWORK:
  INT_1 — Western entry/exit, 2-phase signal
  INT_2 — Central junction, main bottleneck, 3-phase signal
  INT_3 — Southern branch entry/exit, 2-phase signal

AVAILABLE TOOLS AND WHEN TO USE THEM:
  get_intersection_status    — status, queues, wait times for one intersection
  get_signal_history         — signal phase history for one intersection
  explain_congestion         — what happened at one intersection at a specific hour
  get_agent_comparison       — fixed timer vs adaptive agent results (use for "did agent help?")
  get_flow_analysis          — analysis of a specific traffic flow (FLOW_1 to FLOW_6)
  get_rush_hour_summary      — morning and evening rush hour data (use for "morning commute" questions)
  get_worst_approaches       — top N worst approaches network-wide (use for "worst bottleneck" questions)
  get_network_summary        — complete network overview (use for "full summary" or broad questions)

KEY RESULTS:
  Adaptive ML agent vs fixed timer:
    Average wait time: -9.7%
    Maximum wait time: -32.4%
    Additional cars served: +374
  INT_2 is the bottleneck — all 6 traffic flows pass through it
  Morning rush: 7-9am | Evening rush: 4-7pm (hours 16-19)
  All wait times are in SECONDS not minutes

RESPONSE STYLE:
  - 2-4 plain conversational sentences
  - Always cite specific numbers from tool results
  - No bullet points unless explicitly asked
  - Speak like explaining to a non-technical person
  - For human questions like "was traffic bad?" translate to data and answer plainly
'''

# words that indicate the user is asking about traffic
TRAFFIC_KEYWORDS = [
    'int_1', 'int_2', 'int_3', 'intersection', 'wait', 'queue',
    'congestion', 'signal', 'traffic', 'status', 'approach', 'phase',
    'rush', 'agent', 'performance', 'compare', 'worst', 'best',
    'hour', 'cars', 'served', 'summary', 'slow', 'busy', 'fast'
]

SMALL_TALK = {
    'hi', 'hello', 'hey', 'thanks', 'thank you', 'ok', 'okay',
    'got it', 'understood', 'good', 'great', 'nice', 'cool',
    'ok bye', 'okay bye', 'see you', 'later', 'good morning',
    'good evening', 'good night', 'sounds good', 'perfect'
}

SMALL_TALK_RESPONSES = {
    'hi'           : 'Hi! Ask me anything about the traffic network.',
    'hello'        : 'Hello! What would you like to know about the intersections?',
    'hey'          : 'Hey! Feel free to ask about INT_1, INT_2, or INT_3.',
    'thanks'       : 'You\'re welcome!',
    'thank you'    : 'Happy to help!',
    'ok'           : 'Sure — what else would you like to know?',
    'okay'         : 'Sure — what else would you like to know?',
    'got it'       : 'Great — anything else?',
    'understood'   : 'Great — anything else?',
    'good'         : 'Glad to hear it! Anything else?',
    'great'        : 'Glad to hear it! Anything else?',
    'cool'         : 'Glad to help! Anything else?',
    'sounds good'  : 'Great — anything else?',
    'perfect'      : 'Glad to help! Anything else?',
    'ok bye'       : 'Goodbye! Come back anytime.',
    'okay bye'     : 'Goodbye! Come back anytime.',
    'see you'      : 'See you later!',
    'good morning' : 'Good morning! Ask me about the traffic network.',
    'good evening' : 'Good evening! Ask me about the traffic network.',
    'good night'   : 'Good night!',
}

def get_rag_context(question):
    '''
    Retrieves relevant policy sections for the user's question.
    Injected into the system prompt so the LLM can cite policy rules.
    Returns empty string if retrieval fails — graceful degradation.
    '''
    try:
        context = retrieve(question, top_k=2)
        return f'\n\nRELEVANT POLICY SECTIONS:\n{context}'
    except Exception:
        return ''

def is_small_talk(text):
    '''Returns True if message is conversational, not a traffic question.'''
    normalized = text.lower().strip().rstrip('!?.')
    if normalized in SMALL_TALK:
        return True
    has_traffic = any(kw in text.lower() for kw in TRAFFIC_KEYWORDS)
    if len(text.split()) <= 3 and not has_traffic:
        return True
    return False


def run_agentic_loop(client, messages):
    '''
    Runs the tool-calling loop until the model gives a final response.
    Groq supports native tool_calls — no manual parsing needed.
    The model automatically decides which tools to call and when to stop.

    Returns the final response text.
    '''
    while True:
        response = client.chat.completions.create(
            model       = 'llama-3.3-70b-versatile',
            messages    = messages,
            tools       = GROQ_TOOLS,
            tool_choice = 'auto',
            max_tokens  = 1024,
            temperature = 0.3    # lower = more consistent, less creative
        )

        message       = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        messages.append(message)

        if finish_reason == 'stop':
            # model gave final answer
            return message.content

        elif finish_reason == 'tool_calls':
            # model wants to call tools — dispatch all of them
            for tool_call in message.tool_calls:
                tool_name  = tool_call.function.name
                tool_input = json.loads(tool_call.function.arguments)

                print(f'  [calling {tool_name}...]')
                result = dispatch_tool(tool_name, tool_input)

                # truncate very long results to avoid context overflow
                # 2000 chars is enough for the model to get key data
                if len(result) > 2000:
                    result = result[:2000] + '\n...(truncated for context length)'

                # send result back to model
                messages.append({
                    'role'        : 'tool',
                    'tool_call_id': tool_call.id,
                    'content'     : result
                })

        else:
            return f'Unexpected finish reason: {finish_reason}'


def run_conversation():
    '''
    Main conversation loop using Groq + LLaMA 3.1 70B.
    Handles small talk, single tools, and multi-tool questions.
    Near-instant responses via Groq's fast inference.
    '''
    client   = Groq()
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]

    print('=' * 55)
    print('TrafficPulse Assistant (Groq — LLaMA 3.1 70B)')
    print('=' * 55)
    print('Ask about the traffic network. Type "quit" to exit.')
    print('=' * 55)
    print()

    while True:
        try:
            user_input = input('You: ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nGoodbye!')
            break

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
            print('Goodbye!')
            break

        # handle small talk without calling the LLM
        normalized = user_input.lower().strip().rstrip('!?.')
        if is_small_talk(user_input):
            reply = SMALL_TALK_RESPONSES.get(
                normalized,
                'Feel free to ask anything about the traffic network!'
            )
            messages.append({'role': 'user',      'content': user_input})
            messages.append({'role': 'assistant', 'content': reply})
            print(f'\nAssistant: {reply}\n')
            continue

        # # traffic question — run agentic loop
        # messages.append({'role': 'user', 'content': user_input})
        # print('  [thinking...]')

        # response = run_agentic_loop(client, messages)
        # print(f'\nAssistant: {response}\n')

        # traffic question — retrieve relevant policy + run agentic loop
        rag_context = get_rag_context(user_input)
        full_input  = user_input + rag_context

        messages.append({'role': 'user', 'content': full_input})
        print('  [thinking...]')

        response = run_agentic_loop(client, messages)
        print(f'\nAssistant: {response}\n')


def ask_once(question: str) -> str:
    '''
    Single-question interface for Streamlit dashboard on Day 9.
    Identical interface to assistant.py so dashboard works with both.
    '''
    if is_small_talk(question):
        normalized = question.lower().strip().rstrip('!?.')
        return SMALL_TALK_RESPONSES.get(
            normalized,
            'Feel free to ask anything about the traffic network!'
        )

    client   = Groq()
    rag_context = get_rag_context(question)
    full_input = question + rag_context
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user',   'content': question}
    ]

    return run_agentic_loop(client, messages)


if __name__ == '__main__':
    run_conversation()