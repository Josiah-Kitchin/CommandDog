
from openai import OpenAI
import speech
import json
from command_scripts import DogController

dog = DogController()

def main(): 
    client = OpenAI()

    print("Recording...")
    audio = speech.record(duration=3)
    text = speech.transcribe(audio)

    print(f"You said: ${text}")

    messages = [{"role": "user", "content": f"{text}"}]
    tools = [{
    'type': 'function',
    'function': {  
        'name': 'flip',
        'description': 'flip the dog in the chosen direction',
        'parameters': {
            'type': 'object', 
            'properties': {
                'direction': {
                    'type': 'string',
                    'description': 'A flipping direction for the dog. Must be either front, left, back, or right'
                }
            },
            'required': ['direction'], # 'required' is an attribute of the parameters schema
            'additionalProperties': False # This is also an attribute of the parameters schema
        }
    }
    }]
    
    try:
        response = client.chat.completions.create(
            model = "gpt-4.1-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        response = response.choices[0].message 
        tool_calls = response.tool_calls

        if tool_calls:
            for call in tool_calls:
                function_name = call.function.name
                function_args_str = call.function.arguments # Arguments are a JSON string
                function_args = json.loads(function_args_str)

                print(f"Function to call: {function_name}")
                print(f"Arguments: {function_args}")

                if function_name == 'flip':
                    direction_value = function_args.get('direction')
                    if direction_value is not None:
                            # Call your Python function
                            function_response_content = dog.flip(direction_value)
                            messages.append(response)
                            messages.append({
                                "tool_call_id": tool_calls.id,
                                "role": "tool",
                                "name": function_name,
                                "content": function_response_content,
                            })

                            # Now get a new response from the LLM using the tool's output
                            print("\nSending function result back to OpenAI...")
                            second_response = client.chat.completions.create(
                                model="gpt-4o",
                                messages=messages
                            )
                            final_answer = second_response.choices[0].message.content
                            print(f"\nLLM's final answer after tool use: {final_answer}")
                            # --- End of "Next Step" ---
    except Exception as e:
        print(f"An error occurred: {e}")










if __name__ == "__main__": 
    main()
