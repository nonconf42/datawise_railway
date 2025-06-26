# from openai import OpenAI
# import os

# class LLM:
#     def __init__(self, platform):
#         self.platform = platform
#         if platform == "openai":
#             api_key = os.environ.get('OPENAI_API_KEY')
#             self.llm_model = OpenAI(api_key=api_key)
#         elif platform == 'deepinfra':
#             api_key = os.environ.get('DEEPINFRA_API_KEY')
#             self.llm_model = OpenAI(
#                 base_url="https://api.deepinfra.com/v1/openai",
#                 api_key=api_key
#             )
     

#     def llm_call(self, prompt, model_name="gpt-4o"):
#         if self.platform == "openai":
#             output_text = self.llm_model.chat.completions.create(
#                 model=model_name,
#                 messages=[
#                     {"role": "system", "content": "You are a helpful assistant."},
#                     {"role": "user", "content": prompt}
#                 ]
#             )
                
#             output_text = output_text.choices[0].message.content
        
#         elif self.platform == 'deepinfra':
#             output_text = self.llm_model.chat.completions.create(
#                 model=model_name,
#                 messages=[{"role": "user", "content": prompt}],
#             )
#             output_text = output_text.choices[0].message.content


        
#         return output_text



import os
from openai import OpenAI
from anthropic import Anthropic

# class LLM:
#     def __init__(self, platform):
#         self.platform = platform
#         if platform == "openai":
#             api_key = os.environ.get('OPENAI_API_KEY')
#             self.llm_model = OpenAI(api_key=api_key)
#         elif platform == 'deepinfra':
#             api_key = os.environ.get('DEEPINFRA_API_KEY')
#             self.llm_model = OpenAI(
#                 base_url="https://api.deepinfra.com/v1/openai",
#                 api_key=api_key
#             )
#         elif platform == 'anthropic':
#             api_key = os.environ.get('ANTHROPIC_API_KEY')
#             self.llm_model = Anthropic(api_key=api_key)
#         else:
#             raise ValueError(f"Unsupported platform: {platform}")

#     def llm_call(self, prompt, model_name="gpt-4o"):
#         if self.platform == "openai":
#             output_text = self.llm_model.chat.completions.create(
#                 model=model_name,
#                 messages=[
#                     {"role": "system", "content": "You are a helpful assistant."},
#                     {"role": "user", "content": prompt}
#                 ]
#             )
#             output_text = output_text.choices[0].message.content

#         elif self.platform == 'deepinfra':
#             output_text = self.llm_model.chat.completions.create(
#                 model=model_name,
#                 messages=[{"role": "user", "content": prompt}],
#             )
#             output_text = output_text.choices[0].message.content

#         elif self.platform == 'anthropic':
#             output_text = self.llm_model.messages.create(
#                 model=model_name,
#                 messages=[{
#                     "role": "user",
#                     "content": prompt
#                 }],
#                 max_tokens=20000
#             )
#             output_text = output_text.content[0].text

#         return output_text


import os
from openai import OpenAI
from anthropic import Anthropic

class LLM:
    # Default model names for each platform
    DEFAULT_MODELS = {
        "openai": "gpt-4o",
        "deepinfra": "mistral-7b-instruct",
        "anthropic": "claude-3-5-sonnet-20241022"
    }

    def __init__(self, platform):
        self.platform = platform
        if platform == "openai":
            api_key = os.environ.get('OPENAI_API_KEY')
            self.llm_model = OpenAI(api_key=api_key)
        elif platform == 'deepinfra':
            api_key = os.environ.get('DEEPINFRA_API_KEY')
            self.llm_model = OpenAI(
                base_url="https://api.deepinfra.com/v1/openai",
                api_key=api_key
            )
        elif platform == 'anthropic':
            os.environ['ANTHROPIC_API_KEY'] = "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"
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            self.llm_model = Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unsupported platform: {platform}")

    def llm_call(self, prompt, model_name=None):
        # Use platform-specific default model if none provided
        if model_name is None:
            model_name = self.DEFAULT_MODELS.get(self.platform)

        if self.platform == "openai":
            output_text = self.llm_model.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            output_text = output_text.choices[0].message.content

        elif self.platform == 'deepinfra':
            output_text = self.llm_model.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            output_text = output_text.choices[0].message.content

        elif self.platform == 'anthropic':
            output_text = self.llm_model.messages.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                max_tokens=8192,  
            )
            output_text = output_text.content[0].text

        return output_text