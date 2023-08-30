# %pip install --upgrade openai
# %pip install stability-sdk

import openai
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from PIL import Image
from stability_sdk import client

import os
import io

openai.api_key = "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"
os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
os.environ['STABILITY_KEY'] = 'sk-yCtTlzutKbEYT1NJbF6fCjWEJw0oHSSlVtSZxSWl76oepBtT'

def gen_sentences_and_prompts(gpt_model, topic, tasks, tasks_remark, number):
  image_guideline = """Here are some guidelines for generating high-quality images. You should write the prompts following these guidelines. \
1 - Be specific and detailed about the image. \
2 - Use sensory language to describe textures, sounds, smells, and other sensory details. \
3 - Consider composition and perspective by providing guidelines on camera angles, framing, and object arrangement. \
4 - Set the mood and atmosphere by describing the desired emotional context of the image. \
5 - Incorporate storytelling elements to give the AI image a sense of narrative or drama. \
6 - Reference existing art or images as inspiration or guidance for the desired aesthetic."""
  system_message = """You are a knowledgable assistant. You need to write sentences using the best of your knowledge."""
  user_message = f"""Your job is to generate {number} sentences in the topic and tasks delimited by triple backticks. \
```Topic: {topic}. Tasks: {tasks}``` \
Make sure that you must satisfy the requirements delimited by <>. \
<Requirements: {tasks_remark}> \
For each sentence, write a prompt for generating its corresponding images. \
{image_guideline} \
The prompts must include information about the topic. \
The Output must be {number} sentences and {number} prompts. \
Output it as a Json object, where the key is the sentence and the value is the prompt."""

  contents_response = openai.ChatCompletion.create(
      model=gpt_model,
      messages=[
          {"role": "system", "content": system_message},
          {"role": "user", "content": user_message},
      ],
      temperature=0,
  )
  contents_response_text = contents_response["choices"][0]["message"]["content"]
  contents_response_text = eval(contents_response_text)

  sentences = []
  prompts = []
  for k,v in contents_response_text.items():
    sentences.append(k)
    prompts.append(v)
  
  return sentences, prompts



def gen_slogan(gpt_model, topic, sentences):
  system_message = "I will give you a topic and some related sentences. Your job is to generate a slogan about them."
  user_message = "The topic is: " + topic + ". The sentences are " + " ".join(sentences)

  slogan_response = openai.ChatCompletion.create(
      model=gpt_model,
      messages=[
          {"role": "system", "content": system_message},
          {"role": "user", "content": user_message},
      ],
      temperature=0,
  )
  slogan_response_text = slogan_response["choices"][0]["message"]["content"]

  return slogan_response_text


def gen_background_prompt(gpt_model, topic, slogan):
  system_message = "I will give you a topic and a slogan. You will need to write a prompt for generating AI Images. The prompt should specify a style."
  user_message = "The topic is: " + topic + ". The slogan is: " + slogan

  background_prompt_response = openai.ChatCompletion.create(
      model=gpt_model,
      messages=[
          {"role": "system", "content": system_message},
          {"role": "user", "content": user_message},
      ],
      temperature=0,
  )
  background_prompt_response_text = background_prompt_response["choices"][0]["message"]["content"]
  return background_prompt_response_text



def gen_images(prompts, quality, image_style):
  if quality == "low":
    engine_type = "stable-diffusion-v1"
    n_steps = 15
  elif quality == "medium":
    engine_type = "stable-diffusion-v1"
    n_steps = 30
  elif quality == "high":
    engine_type = "stable-diffusion-512-v2-0"
    n_steps = 20

  stability_api = client.StabilityInference(
      key=os.environ['STABILITY_KEY'],  # API Key reference.
      verbose=True,  # Print debug messages.
      engine=engine_type #"stable-diffusion-v1",  # Set the engine to use for generation.
      # Available engines: stable-diffusion-v1 stable-diffusion-v1-5 stable-diffusion-512-v2-0 stable-diffusion-768-v2-0
      # stable-diffusion-512-v2-1 stable-diffusion-768-v2-1 stable-diffusion-xl-beta-v2-2-2 stable-inpainting-v1-0 stable-inpainting-512-v2-0
  )

  images = []


  for i, prompt in enumerate(prompts):
    answers = stability_api.generate(
      prompt=prompt,
      style_preset = image_style,
      seed=992446758,  # If a seed is provided, the resulting generated image will be deterministic.
      # What this means is that as long as all generation parameters remain the same, you can always recall the same image simply by generating it again.
      # Note: This isn't quite the case for CLIP Guided generations, which we tackle in the CLIP Guidance documentation.
      steps=n_steps,  # Amount of inference steps performed on image generation. Defaults to 30.
      cfg_scale=8.0,  # Influences how strongly your generation is guided to match your prompt.
      # Setting this value higher increases the strength in which it tries to match your prompt.
      # Defaults to 7.0 if not specified.
      width=512,  # Generation width, defaults to 512 if not included.
      height=512,  # Generation height, defaults to 512 if not included.
      samples=1,  # Number of images to generate, defaults to 1 if not included.
      sampler=generation.SAMPLER_K_DPMPP_2M  # Choose which sampler we want to denoise our generation with.
      # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
      # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m, k_dpmpp_sde)
    )

    for resp in answers:
      for artifact in resp.artifacts:
        if artifact.finish_reason == generation.FILTER:
          print("Generation failed.")
        if artifact.type == generation.ARTIFACT_IMAGE:
          img = Image.open(io.BytesIO(artifact.binary))
          img.save(str(i)+".png")
          images.append(str(i)+".png")

  return images



def gen_background(background_prompt):
  stability_api = client.StabilityInference(
      key=os.environ['STABILITY_KEY'],  # API Key reference.
      verbose=True,  # Print debug messages.
      engine="stable-diffusion-xl-beta-v2-2-2",  # Set the engine to use for generation.
      # Available engines: stable-diffusion-v1 stable-diffusion-v1-5 stable-diffusion-512-v2-0 stable-diffusion-768-v2-0
      # stable-diffusion-512-v2-1 stable-diffusion-768-v2-1 stable-diffusion-xl-beta-v2-2-2 stable-inpainting-v1-0 stable-inpainting-512-v2-0
  )


  answers = stability_api.generate(
    prompt=background_prompt,
    seed=12345,  # If a seed is provided, the resulting generated image will be deterministic.
    # What this means is that as long as all generation parameters remain the same, you can always recall the same image simply by generating it again.
    # Note: This isn't quite the case for CLIP Guided generations, which we tackle in the CLIP Guidance documentation.
    steps=30,  # Amount of inference steps performed on image generation. Defaults to 30.
    cfg_scale=8.0,  # Influences how strongly your generation is guided to match your prompt.
    # Setting this value higher increases the strength in which it tries to match your prompt.
    # Defaults to 7.0 if not specified.
    width=512,  # Generation width, defaults to 512 if not included.
    height=256,  # Generation height, defaults to 512 if not included.
    samples=1,  # Number of images to generate, defaults to 1 if not included.
    sampler=generation.SAMPLER_K_DPMPP_2M  # Choose which sampler we want to denoise our generation with.
    # Defaults to k_dpmpp_2m if not specified. Clip Guidance only supports ancestral samplers.
    # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m, k_dpmpp_sde)
  )

  for resp in answers:
    for artifact in resp.artifacts:
      if artifact.finish_reason == generation.FILTER:
        print("Generation failed.")
      if artifact.type == generation.ARTIFACT_IMAGE:
        img = Image.open(io.BytesIO(artifact.binary))
        img.save("background.png")


  return "background.png"


def gen_html(gpt_model, topic, sentences, slogan, images, background, website_demand):
  system_message = "I will give you a topic, some related sentences, the corresponding images and a slogan. Your job is to generate a html file. "
  system_message += "The header contains the topic only. "
  system_message += "A container contains the slogan and a background image. The source of the background image is" + str(background) + ". "
  system_message += "The background image is not the background for the whole html page. It is only local for the slogan container. It should somehow transparent, so that the user can read the slogan. "
  system_message += "The text color of the slogan should be constract color of the background image. "
  system_message += "In addtion, another container shows a list of images and their corresponding sentences. The source of the images are " + str(images)
  system_message += website_demand + "a right set type "
  user_message = "The topic is: " + topic + ". The sentences are " + " ".join(sentences) + " The slogan is " + slogan


  html_response = openai.ChatCompletion.create(
      model=gpt_model,
      messages=[
          {"role": "system", "content": system_message},
          {"role": "user", "content": user_message},
      ],
      temperature=0,
  )

  html_response_text = html_response["choices"][0]["message"]["content"]

  return html_response_text







def gen_website(gpt_model, topic, tasks, tasks_remark, number, quality, image_style, website_demand):
  print("generating sentences and prompts...")
  sentences, prompts = gen_sentences_and_prompts(gpt_model, topic, tasks, tasks_remark, number)
  print("generating slogan...")
  slogan = gen_slogan(gpt_model, topic, sentences)
  print("generating background prompts...")
  background_prompt = gen_background_prompt(gpt_model, topic, slogan)
  print("generating images...")
  images = gen_images(prompts, quality, image_style)
  print("generating background images...")
  background = gen_background(background_prompt)
  print("generating html...")
  html_file = gen_html(gpt_model, topic, sentences, slogan, images, background, website_demand)

  f = open("website.html", "w")
  f.write(html_file)
  f.close()


if __name__ == '__main__':
  gpt_model = "gpt-3.5-turbo"
  topic = "motorcycle"
  topic_demand = "Introduce the motorcycle."
  userdemandremark = "Introduce 3 motorcycle. Motorcycle brands should compare performance, price, and usage. Examples of uses include whether it is for running a race track or riding a motorcycle."
  number = 3
  quality = "high"
  image_style = "3d-model"
  website_demand = "The background color of the html must be in a dark theme."
  gen_website(gpt_model, topic, topic_demand, userdemandremark, number, quality, image_style, website_demand)
