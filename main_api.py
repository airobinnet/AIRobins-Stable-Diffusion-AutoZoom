import base64
from io import BytesIO
import io
import replicate
import os
from PIL import Image
import imageio
import numpy as np
import tqdm
import math
import tempfile
import requests
import cv2
import random
import string

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)


# Set your API token
os.environ['REPLICATE_API_TOKEN'] = "YOUR_REPLICATE_API_TOKEN"

# Open the image and mask files
image_path = "start.png"
prompt = ""

counter = 1

frame_count: int = 4
target: str = 'test'
frames: int = 60
fps: float = 60

init_msg = """
╔═╗╦╦═╗┌─┐┌┐ ┬┌┐┌┌─┐  ╔═╗╔╦╗  ╔═╗┬ ┬┌┬┐┌─┐╔═╗┌─┐┌─┐┌┬┐  
╠═╣║╠╦╝│ │├┴┐││││└─┐  ╚═╗ ║║  ╠═╣│ │ │ │ │╔═╝│ ││ ││││  
╩ ╩╩╩╚═└─┘└─┘┴┘└┘└─┘  ╚═╝═╩╝  ╩ ╩└─┘ ┴ └─┘╚═╝└─┘└─┘┴ ┴  
version 0.4.2                              airobin.net
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html', show_dropdown=False)

@app.route('/inpaint', methods=['POST'])
def inpaint():
    if request.method == 'POST':
        data = request.get_json()
        image_data = data.get('cropped_image')
        image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        image = BytesIO(image_bytes)
        new_prompt = data.get('new_prompt')
        inference_num = int(data.get('num_inference'))
        target = data.get('target')
        amount = int(data.get('amount'))
        number = int(data.get('number'))
        # generate a random target name
        if target == "":
            target = ''.join(random.choice(string.ascii_lowercase) for i in range(10))

        # Call the do_magic function with the new prompt
        new_image_paths = zoom_out_inpaint(image, target, new_prompt, inference_num, amount, number)

        # Return the new image paths
        return jsonify(new_image_paths)

@app.route('/try_again', methods=['POST'])
def try_again():
    # Get the image path and prompt from the request
    image_path = request.form.get('image_path')
    prompt = request.form.get('prompt')
    amount = int(request.form.get('amount'))
    number = int(request.form.get('number'))
    inference_num = int(request.form.get('num_inference'))
    # generate a random target name
    if target == "":
        target = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
    # Call the do_magic function with the same prompt
    new_image_paths = zoom_out_inpaint(image_path, target, prompt, inference_num, amount, number)

    # Return the new image path
    return new_image_paths[0]

@app.route('/make_video', methods=['POST'])
def make_video():
    # Get the image paths from the request
    image_paths = request.form.getlist('image_paths[]')

    target = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
    # Call the animate function with the image_paths
    video_path = animate(len(image_paths) - 1, frames, fps, target, image_paths)

    # Return the video path
    return video_path


def outpaint(image_path, mask_path, prompt, target, inference_num, amount=1, number=0):
    global counter
    image_paths = []
    # Run the prediction
    iterator = replicate.run(
    "stability-ai/stable-diffusion-inpainting:c28b92a7ecd66eee4aefcd8a94eb9e7f6c3805d5f06038165407fb5cb355ba67",
    input={
        "prompt": prompt,
        "image": open(image_path, 'rb'),
        "mask": open(mask_path, 'rb'),
        "num_inference_steps": inference_num,
        "guidance_scale": 7.6,
        "num_outputs" : amount,
        "negative_prompt": "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad proportions, cloned face, deformed, dehydrated, disfigured, duplicate, extra arms, extra fingers, extra legs, extra limbs, fused fingers, gross proportions, long neck, malformed limbs, missing arms, missing legs, morbid, mutated hands, mutation, mutilated, out of frame, poorly drawn face, poorly drawn hands, too many fingers, ugly"
    },
    )
    
    # Print the results
    for image_url in iterator:
        number += 1
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))

        # Open the input image and resize it to match the output image size
        input_img = Image.open(image_path).resize(img.size)

        # Paste the input image onto the output image
        img.paste(input_img, (0, 0), input_img)
        print("Saving frame " + str(number) + "...\n with prompt: " + prompt + "\nand inference_num: " + str(inference_num) + "\n")
        local_image_path = "static/" + target + "/frame" + str(number) + ".png"
        img.save(local_image_path)
        counter += 1
        image_paths.append(local_image_path)
    return image_paths

def create_mask(input_image_path):
    input_image = Image.open(input_image_path)
    mask = Image.new("L", input_image.size, 255)
    input_pixels = input_image.load()
    mask_pixels = mask.load()
    width, height = input_image.size
    for y in range(height):
        for x in range(width):
            r, g, b, a = input_pixels[x, y]
            if a == 0:
                mask_pixels[x, y] = 255
            else:
                mask_pixels[x, y] = 0
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_mask:
            mask.save(temp_mask.name)
            return temp_mask.name

def zoom_out(input_image_path):
    img = Image.open(input_image_path)
    w, h = img.size
    img = img.resize((w // 2, h // 2))  # resize the image to half its size
    img_out = Image.new("RGBA", (w, h), (255, 255, 255, 0))  # create a new blank image with the original size
    img_out.paste(img, (w // 4, h // 4))  # paste the resized image into the center of the new image

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
        img_out.save(temp_img.name)
        return temp_img.name

def zoom_at(img, zoom, x, y):
    w, h = img.size

    # Expand the region to zoom out
    xz = x * zoom / 2
    yz = y * zoom / 2
    box = (int(max(0, x - xz)), int(max(0, y - yz)), int(min(w, x + xz)), int(min(h, y + yz)))
    cropped_img = img.crop(box)

    # Resize the cropped image back to the original size using OpenCV
    cropped_img_cv = np.array(cropped_img)
    # Change the interpolation method to INTER_CUBIC or INTER_LANCZOS4 for a sharper image
    resized_img_cv = cv2.resize(cropped_img_cv, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # Convert the resized image back to PIL image
    resized_img = Image.fromarray(resized_img_cv)

    return resized_img

def animate(frame_count, frames, fps, target, image_base64s):
    log_scale = math.log(2)
    with imageio.get_writer('static/' + target + ".mp4", mode='I', fps=fps) as writer:
        for i in range(frame_count + 1):
            base64_img = image_base64s[i]
            if 'base64,' in base64_img:
                base64_img = base64_img.split('base64,')[1]
            # Add the missing padding to the base64 string
            padding = 4 - len(base64_img) % 4
            if padding:
                base64_img += "=" * padding
            img_data = base64.b64decode(base64_img)
            img = Image.open(io.BytesIO(img_data))
            w, h = img.size
            for frame in tqdm.tqdm(range(frames)):
                if i == 0:  # no zoom for the first frame
                    zoomed = img
                else:
                    zoom = math.exp(log_scale * ((frame-1) / (frames-1))) if frame > 0 else 1
                    zoomed = zoom_at(img, zoom, w // 2, h // 2)
                zoomed = zoomed.convert('RGBA')
                background = Image.new('RGB', zoomed.size, (255, 255, 255))
                background.paste(zoomed, mask=zoomed.split()[3])  # 3 is the alpha channel
                writer.append_data(np.array(background))
            if i == frame_count:
                for _ in range(fps):
                    writer.append_data(np.array(background))

        # Reverse the zoom process
        for i in range(frame_count, -1, -1):
            base64_img = image_base64s[i]
            if 'base64,' in base64_img:
                base64_img = base64_img.split('base64,')[1]
            # Add the missing padding to the base64 string
            padding = 4 - len(base64_img) % 4
            if padding:
                base64_img += "=" * padding
            img_data = base64.b64decode(base64_img)
            img = Image.open(io.BytesIO(img_data))
            w, h = img.size
            for frame in tqdm.tqdm(range(frames-1, -1, -1)):
                if i == 0:  # no zoom for the last frame
                    zoomed = img
                else:
                    zoom = math.exp(log_scale * ((frame-1) / (frames-1))) if frame > 0 else 1
                    zoomed = zoom_at(img, zoom, w // 2, h // 2)
                zoomed = zoomed.convert('RGBA')
                background = Image.new('RGB', zoomed.size, (255, 255, 255))
                background.paste(zoomed, mask=zoomed.split()[3])  # 3 is the alpha channel
                writer.append_data(np.array(background))

    return "static/" + target + ".mp4"

def zoom_out_inpaint(image_path, target, prompt, inference_num, amount, number):
    # Ensure the directory exists
    os.makedirs("static/" + target, exist_ok=True)
    Image.open(image_path).save("static/" + target + "/frame" + str(number) + ".png")
    print(f"Processing frame {str(number)}...")
    image_path = zoom_out(image_path)
    mask_path = create_mask(image_path)
    image_path = outpaint(image_path, mask_path, prompt, target, inference_num, amount, number)
    #result = animate(frame_count, target, frames, fps)
    return image_path

if __name__ == '__main__':
    print(init_msg)
    app.run(debug=False, port=5000, host='0.0.0.0')