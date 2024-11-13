from openai import OpenAI
import os
import time
import json
import datetime

class LLMPrompter():
    def __init__(self, gpt_version, api_key, base_url) -> None:
        self.gpt_version = gpt_version
        if api_key is None:
            raise ValueError("OpenAI API key is not provided.")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url if base_url else "https://api.openai.com/v1"
        )

    def query(self, prompt: str, sampling_params: dict, save: bool, save_dir: str) -> str:
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.gpt_version,
                    messages=[
                        {"role": "system", "content": prompt['system']},
                        {"role": "user", "content": prompt['user']},
                    ],
                    **sampling_params
                )
            except Exception as e:
                print("Request failed, sleep 2 secs and try again...", e)
                time.sleep(2)
                continue
            break

        if save:
            key = self.make_key()
            output = {}
            os.system('mkdir -p {}'.format(save_dir))
            if os.path.exists(os.path.join(save_dir, 'response.json')):
                with open(os.path.join(save_dir, 'response.json'), 'r') as f:
                    prev_response = json.load(f)
                    output = prev_response

            with open(os.path.join(save_dir, 'response.json'), 'w') as f:
                output[key] = {
                    'prompt': prompt,
                    'sampling_params': sampling_params,
                    'response': response.choices[0].message.content.strip()
                }
                json.dump(output, f, indent=4)
        
        return response.choices[0].message.content.strip(), None

    def make_key(self):
        return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")