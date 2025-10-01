from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()

# Use NVIDIA's endpoint but with OpenAI client
client = OpenAI(
    api_key=os.getenv("api_key"),  # your NVIDIA API key
    base_url="https://integrate.api.nvidia.com/v1"  # point to NVIDIA
)

# Make a chat completion call to Llama
response = client.chat.completions.create(
    model="meta/llama-4-maverick-17b-128e-instruct",  # NVIDIA's Llama model
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain the difference between Groq, NVIDIA, and AWS Bedrock."}
    ],
    max_tokens=512,
    temperature=1.0,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stream=False
)

print(response.choices[0].message.content)
