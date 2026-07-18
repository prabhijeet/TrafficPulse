# llm/assistant.py

'''
Day 7 — LLM Assistant with Tool-Calling using Ollama + Mistral 7B.
Runs entirely locally — no API key, no internet, no cost.

Final version fixes:
  - Handles TOOL_CALL and TOOL_CALL N: formats from Mistral
  - Strips response prefixes (A:, Assistant:, User:)
  - Multiple tool calls dispatched correctly
  - Real data always injected before final answer
'''

import os
import json
import re
import ollama
from llm.tools import dispatch_tool

SYSTEM_PROMPT = '''You are TrafficPulse, a traffic assistant for a 3-intersection urban network.

You have exactly 3 tools. Use ONLY these — never invent other tool names.

TOOL 1: get_intersection_status
  Use for: status, wait times, queue lengths, performance of an intersection
  Format: TOOL_CALL: {"tool": "get_intersection_status", "args": {"intersection_id": "INT_1"}}

TOOL 2: get_signal_history  
  Use for: signal phases, timing, how often each phase ran
  Format: TOOL_CALL: {"tool": "get_signal_history", "args": {"intersection_id": "INT_2", "last_n_hours": 2}}

TOOL 3: explain_congestion
  Use for: why congestion happened at a specific hour
  Format: TOOL_CALL: {"tool": "explain_congestion", "args": {"intersection_id": "INT_2", "hour": 8}}

CRITICAL RULES:
- Always start with TOOL_CALL: (never TOOL_CALL 1: or variations)
- For comparisons, call the tool once per intersection on separate lines
- All wait times in tool results are in SECONDS — never say minutes
- After getting real data, answer in 2-4 plain conversational sentences
- Cite exact numbers from tool results
- Never start response with User: or Assistant:
- Never make up numbers — only use what the tools return

NETWORK FACTS:
  INT_1: western entry, 2-phase signal
  INT_2: central junction, main bottleneck, 3-phase signal
  INT_3: southern branch, 2-phase signal
  Adaptive ML agent vs fixed timer: avg wait -9.7%, max wait -32.4%, +374 more cars served
'''


def parse_all_tool_calls(text):
    '''
    Extracts ALL TOOL_CALLs from Mistral response.
    Handles variations: TOOL_CALL:, TOOL_CALL 1:, TOOL_CALL 2: etc.
    Uses brace counting to correctly parse nested JSON.
    '''
    results = []

    # find all positions of TOOL_CALL (with or without number suffix)
    for match in re.finditer(r'TOOL_CALL\s*\d*\s*:', text):
        start_brace = text.find('{', match.end())
        if start_brace == -1:
            continue

        # count braces to find matching close
        depth = 0
        end   = start_brace
        for i, ch in enumerate(text[start_brace:], start_brace):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        try:
            raw  = text[start_brace:end]
            call = json.loads(raw)
            name = call.get('tool')
            args = call.get('args', {})
            if 'hour' in args:
                args['hour'] = int(args['hour'])
            if name:
                results.append((name, args))
        except (json.JSONDecodeError, ValueError):
            continue

    return results


def clean_response(text):
    '''
    Strips common Mistral response prefixes that appear
    when it mimics conversation format instead of just answering.
    '''
    # remove lines that are just role labels
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # skip lines that are just "User:", "Assistant:", "A:", etc.
        if re.match(r'^(User|Assistant|A|Q)\s*:', stripped) and len(stripped) < 20:
            continue
        cleaned.append(line)
    text = '\n'.join(cleaned).strip()

    # remove leading role prefix on first line
    text = re.sub(r'^(User|Assistant|A|Q)\s*:\s*', '', text).strip()

    return text

SMALL_TALK = {
    'bye', 'goodbye', 'quit', 'exit', 'thanks', 'thank you',
    'ok', 'okay', 'ok bye', 'okay bye', 'see you', 'later',
    'hi', 'hello', 'hey', 'good morning', 'good evening',
    'yes', 'no', 'sure', 'got it', 'understood'
}

def is_small_talk(text):
    '''
    Returns True if the message is conversational/greeting/farewell
    and doesn't need a tool call.
    '''
    cleaned = text.lower().strip().rstrip('!?.')
    # exact match
    if cleaned in SMALL_TALK:
        return True
    # very short messages with no intersection mention
    if len(cleaned.split()) <= 3 and 'int_' not in cleaned.lower():
        return True
    return False


def run_conversation():
    '''
    Main conversation loop.
    Handles small talk, single tools, and multiple tool calls.
    Always injects real data before generating final answer.
    Skips LLM entirely for greetings and farewells.
    '''
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]

    # simple responses for small talk — no LLM call needed
    small_talk_responses = {
        'hi'          : 'Hi! Ask me anything about the traffic network.',
        'hello'       : 'Hello! What would you like to know about the intersections?',
        'hey'         : 'Hey! Feel free to ask about INT_1, INT_2, or INT_3.',
        'thanks'      : 'You\'re welcome!',
        'thank you'   : 'Happy to help!',
        'ok'          : 'Sure — what else would you like to know?',
        'okay'        : 'Sure — what else would you like to know?',
        'got it'      : 'Great — anything else?',
        'understood'  : 'Great — anything else?',
        'good'        : 'Glad to hear it! Anything else?',
        'great'       : 'Glad to hear it! Anything else?',
        'nice'        : 'Glad to help! Anything else?',
        'ok bye'      : 'Goodbye! Come back anytime.',
        'okay bye'    : 'Goodbye! Come back anytime.',
        'see you'     : 'See you later!',
        'later'       : 'See you later!',
        'good morning': 'Good morning! Ask me about the traffic network.',
        'good evening': 'Good evening! Ask me about the traffic network.',
    }

    print('=' * 55)
    print('TrafficPulse Assistant (Mistral 7B — local)')
    print('=' * 55)
    print('Ask about the traffic network. Type "quit" to exit.')
    print('Responses take 15-40 seconds on Intel Mac.')
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

        # handle exit commands first
        if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
            print('Goodbye!')
            break

        # check for small talk — skip LLM entirely
        normalized = user_input.lower().strip().rstrip('!?.')
        if normalized in small_talk_responses:
            reply = small_talk_responses[normalized]
            messages.append({'role': 'user',      'content': user_input})
            messages.append({'role': 'assistant', 'content': reply})
            print(f'\nAssistant: {reply}\n')
            continue

        # also skip LLM for very short messages with no traffic keywords
        traffic_keywords = ['int_1', 'int_2', 'int_3', 'intersection',
                           'wait', 'queue', 'congestion', 'signal',
                           'traffic', 'status', 'approach', 'phase',
                           'rush', 'agent', 'performance', 'compare',
                           'worst', 'best', 'hour', 'am', 'pm', 'cars']
        has_traffic_keyword = any(
            kw in user_input.lower() for kw in traffic_keywords
        )

        if len(user_input.split()) <= 3 and not has_traffic_keyword:
            reply = 'Could you ask me something specific about the traffic network? For example: "What is the status of INT_2?" or "Why was INT_2 congested at 8am?"'
            messages.append({'role': 'user',      'content': user_input})
            messages.append({'role': 'assistant', 'content': reply})
            print(f'\nAssistant: {reply}\n')
            continue

        # proper traffic question — call Mistral
        messages.append({'role': 'user', 'content': user_input})
        print('  [thinking...]')

        # Step 1 — ask Mistral what tools to use
        response      = ollama.chat(model='mistral', messages=messages)
        assistant_msg = response['message']['content']

        # Step 2 — extract all tool calls
        tool_calls = parse_all_tool_calls(assistant_msg)

        if tool_calls:
            # Step 3 — call each Python function with real data
            all_results = []
            for tool_name, tool_args in tool_calls:
                print(f'  [calling {tool_name}...]')
                tool_result = dispatch_tool(tool_name, tool_args)
                all_results.append(
                    f'Data from {tool_name}:\n{tool_result}'
                )

            combined = '\n\n'.join(all_results)

            # Step 4 — inject real data, ask for clean answer
            follow_up = (
                f'Here is the real data from the tools:\n\n'
                f'{combined}\n\n'
                f'Now answer the original question: "{user_input}"\n'
                f'Rules:\n'
                f'- Use 2-4 plain conversational sentences\n'
                f'- Cite exact numbers from the data\n'
                f'- All wait times are in SECONDS\n'
                f'- Do not start with User: or Assistant:\n'
                f'- No bullet points\n'
                f'- Just answer directly'
            )

            final_messages = [
                {'role': 'system',    'content': SYSTEM_PROMPT},
                {'role': 'user',      'content': user_input},
                {'role': 'assistant', 'content': assistant_msg},
                {'role': 'user',      'content': follow_up}
            ]

            print('  [generating answer...]')
            final_response = ollama.chat(model='mistral', messages=final_messages)
            final_text     = clean_response(final_response['message']['content'])

            messages.append({'role': 'assistant', 'content': final_text})
            print(f'\nAssistant: {final_text}\n')

        else:
            # Mistral answered without tools
            # check if it should have used a tool
            if has_traffic_keyword:
                # force it to try again with stronger instruction
                retry_messages = [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user',   'content': (
                        f'{user_input}\n\n'
                        f'Important: You MUST use a tool to answer this. '
                        f'Start your response with TOOL_CALL:'
                    )}
                ]
                print('  [retrying with tool...]')
                retry_response = ollama.chat(model='mistral', messages=retry_messages)
                retry_msg      = retry_response['message']['content']
                retry_calls    = parse_all_tool_calls(retry_msg)

                if retry_calls:
                    all_results = []
                    for tool_name, tool_args in retry_calls:
                        print(f'  [calling {tool_name}...]')
                        tool_result = dispatch_tool(tool_name, tool_args)
                        all_results.append(
                            f'Data from {tool_name}:\n{tool_result}'
                        )

                    combined  = '\n\n'.join(all_results)
                    follow_up = (
                        f'Real data:\n\n{combined}\n\n'
                        f'Answer "{user_input}" in 2-4 plain sentences. '
                        f'All wait times are in SECONDS. '
                        f'No bullet points. Answer directly.'
                    )
                    final_messages = [
                        {'role': 'system',    'content': SYSTEM_PROMPT},
                        {'role': 'user',      'content': user_input},
                        {'role': 'assistant', 'content': retry_msg},
                        {'role': 'user',      'content': follow_up}
                    ]
                    print('  [generating answer...]')
                    final_response = ollama.chat(model='mistral', messages=final_messages)
                    final_text     = clean_response(
                        final_response['message']['content']
                    )
                    messages.append({'role': 'assistant', 'content': final_text})
                    print(f'\nAssistant: {final_text}\n')
                else:
                    # still no tool — show cleaned response
                    cleaned = clean_response(retry_msg)
                    messages.append({'role': 'assistant', 'content': cleaned})
                    print(f'\nAssistant: {cleaned}\n')
            else:
                cleaned = clean_response(assistant_msg)
                messages.append({'role': 'assistant', 'content': cleaned})
                print(f'\nAssistant: {cleaned}\n')

def ask_once(question: str) -> str:
    '''
    Single-question interface for Streamlit dashboard on Day 9.
    Returns Mistral's response as a plain string.
    '''
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user',   'content': question}
    ]

    response      = ollama.chat(model='mistral', messages=messages)
    assistant_msg = response['message']['content']

    tool_calls = parse_all_tool_calls(assistant_msg)

    if tool_calls:
        all_results = []
        for tool_name, tool_args in tool_calls:
            tool_result = dispatch_tool(tool_name, tool_args)
            all_results.append(f'Data from {tool_name}:\n{tool_result}')

        combined  = '\n\n'.join(all_results)
        follow_up = (
            f'Here is the real data:\n\n{combined}\n\n'
            f'Answer: "{question}"\n'
            f'Use 2-4 plain sentences with specific numbers. '
            f'Answer directly without role-play prefixes.'
        )

        final_messages = [
            {'role': 'system',    'content': SYSTEM_PROMPT},
            {'role': 'user',      'content': question},
            {'role': 'assistant', 'content': assistant_msg},
            {'role': 'user',      'content': follow_up}
        ]

        final = ollama.chat(model='mistral', messages=final_messages)
        return clean_response(final['message']['content'])

    return clean_response(assistant_msg)


if __name__ == '__main__':
    run_conversation()




