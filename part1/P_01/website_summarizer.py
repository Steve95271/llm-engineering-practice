import os
from dotenv import load_dotenv
from part1.Util.scraper import fetch_website_contents
from IPython.display import Markdown, display
from openai import OpenAI

load_dotenv()

# Define key and url
openai = OpenAI(
  api_key = os.environ.get('DEEPSEEK_API_KEY'),
  base_url = "https://api.deepseek.com/v1"
)

system_prompt = """
You are a helpful assistant that summarize the contents of a news website. Identify the advertisement part and ignore it.
Respond in markdown. Do not wrap the markdown in a code block - respond just with the markdown.
"""

user_prompt_prefix = """
Here are the contents of a website.
Provide a short summary of this website.
If it includes news or announcements, then summarize these too.
Website content:
"""

def messages_for(webside):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_prefix + webside}
    ]

def summarize(url):
    website = fetch_website_contents(url)
    response = openai.chat.completions.create(
        model = "deepseek-chat",
        messages = messages_for(website)
    )
    return response.choices[0].message.content

def display_summary(url):
    response_content = summarize(url)
    display(Markdown(response_content))


print(summarize("https://www.bbc.com/news"))
