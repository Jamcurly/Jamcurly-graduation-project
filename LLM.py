from openai import OpenAI
import os
import base64
from dotenv import load_dotenv

load_dotenv()
MODEL="gpt-4o"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def text_content(prompt, MODEL=MODEL, client=client):
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that responds without too much explain!"}, # <-- This is the system message that provides context to the model
            {"role": "user", "content": prompt}  # <-- This is the user message for which the model will generate a response
        ],
        temperature=0.0,
    )
    return completion.choices[0].message.content

def image_content(image_path, prompt, MODEL=MODEL, client=client):
    base64_image = encode_image(image_path)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that responds without too much explain!"},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"}
                }
            ]}
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content

filename = 'figs/manipulator.jpg'
image_fold = os.getcwd()
image_path = os.path.join(image_fold, filename)
prompt = "Determin the degrees of freedom of the embodied entity in the provided image and only provide me with a Python function without explain to model the forward kinematics, \
    and the constant parameters can be determined later.The input of the function should be the free variables."
print(image_content(image_path,prompt))