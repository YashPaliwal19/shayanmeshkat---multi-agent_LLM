from openai import OpenAI
from dotenv import load_dotenv
import os
import re

class LLM_DFA():
    def __init__(self):
        load_dotenv(dotenv_path='../myenv.env')
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.client = client

    def set_open_params(self, model='gpt-3.5-turbo', 
                        temperature=0.7,
                        max_tokens=256,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0):
        
        """ set openai parameters"""
        openai_params = {}    
        openai_params['model'] = model
        openai_params['temperature'] = temperature
        openai_params['max_tokens'] = max_tokens
        openai_params['top_p'] = top_p
        openai_params['frequency_penalty'] = frequency_penalty
        openai_params['presence_penalty'] = presence_penalty
        return openai_params

    def get_completion(self, params, prompt):
        """ GET completion from openai api"""

        # response = openai.Completion.create(
        #     engine = params['model'],
        #     prompt = prompt,
        #     temperature = params['temperature'],
        #     max_tokens = params['max_tokens'],
        #     top_p = params['top_p'],
        #     frequency_penalty = params['frequency_penalty'],
        #     presence_penalty = params['presence_penalty'],
        # )

        response = self.client.chat.completions.create(
            model = params['model'],
            messages = prompt
        )

        return response

    def extract_label_sequence(self, response):
        # Use regular expression to find patterns matching transitions in the response
        # pattern = re.compile(r"(\d+),\s*(\d+),\s*'([^']+)',\s*ConstantRewardFunction\((0|1)\)")
        pattern = re.compile(r"(\d+),\s*(!?\s*\w+),\s*(\d+)")
        matches = pattern.findall(response)

        return matches

    def dfa_extract_dimensions(self, transitions_list):
        num_dimensions = len(transitions_list[0])
        dimension_lists = [[] for _ in range(num_dimensions)]

        for item in transitions_list:
            for i in range(num_dimensions):
                dimension_lists[i].append(item[i])

        return dimension_lists

    def transition_index(self, input_list):
        transition_index_list = []
        for iter in range(len(input_list[0])):
            if input_list[0][iter] != input_list[2][iter]:
                transition_index_list.append(iter)
        return transition_index_list

    def hint_gen_from_dfa(self, output_dfa, transition_index_list):
        hint = ''.join(output_dfa[1][i] for i in transition_index_list)
        return hint


    # params = set_open_params()

