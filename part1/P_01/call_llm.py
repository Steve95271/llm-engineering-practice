import os
from dotenv import load_dotenv
from part1.Util.scraper import fetch_website_contents
from IPython.display import Markdown, display
from openai import OpenAI


load_dotenv()

message = "Hello, deepseek! This is my first ever message to you! Hi!"
messages = [{"role": "user", "content": message}]
print("Prompt: " + message)

# Define key and url
openai = OpenAI(
  api_key = os.environ.get('DEEPSEEK_API_KEY'),
  base_url = "https://api.deepseek.com/v1"
)

# send request
response = openai.chat.completions.create(
  model="deepseek-chat",
  messages=messages
)

print(response.choices[0].message.content)

#
