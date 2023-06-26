import base64
from io import BytesIO
import io
import os
from PIL import Image
import imageio
import numpy as np
import tqdm
import math
import tempfile
import requests
import cv2
from diffusers import StableDiffusionInpaintPipeline
import torch

import random
import string

current_dir = os.getcwd()

cache_dir = os.path.join(current_dir, "cache")

os.makedirs(cache_dir, exist_ok=True)

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

image_path = "start.png"
prompt = ""

counter = 1

frame_count: int = 3
target: str = 'test'
frames: int = 40
fps: float = 30
inference_num = 100

# Global variable to hold the model
global_model = None

init_msg = """
╔═╗╦╦═╗┌─┐┌┐ ┬┌┐┌┌─┐  ╔═╗╔╦╗  ╔═╗┬ ┬┌┬┐┌─┐╔═╗┌─┐┌─┐┌┬┐  
╠═╣║╠╦╝│ │├┴┐││││└─┐  ╚═╗ ║║  ╠═╣│ │ │ │ │╔═╝│ ││ ││││  
╩ ╩╩╩╚═└─┘└─┘┴┘└┘└─┘  ╚═╝═╩╝  ╩ ╩└─┘ ┴ └─┘╚═╝└─┘└─┘┴ ┴  
version 0.4.3                              airobin.net
"""

def load_model():
    """
    Load the pre-trained StableDiffusionInpaintPipeline model from local files and move it to GPU if available.
    Returns the loaded model.
    """
    model = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    model.to("cuda")
    # print out to check if we use GPU
    print("Using GPU: ", torch.cuda.is_available())
    return model

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Render the main index.html template when the user accesses the root path of the web application.
    """
    return render_template('index.html', show_dropdown=True)

@app.route('/inpaint', methods=['POST'])
def inpaint():
    """
    Receive a POST request containing the cropped image, prompt, inference number, target, amount, and number.
    Call the zoom_out_inpaint function with the provided data, and return the new image paths.
    """
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
    """
    Receive a POST request containing the image path, prompt, amount, number, and inference number.
    Call the zoom_out_inpaint function with the same prompt, and return the new image paths.
    """
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
    """
    Receive a POST request containing the list of image paths.
    Call the animate function with the image_paths, and return the video path.
    """
    # Get the image paths from the request
    image_paths = request.form.getlist('image_paths[]')

    target = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
    # Call the animate function with the image_paths
    video_path = animate(len(image_paths) - 1, frames, fps, target, image_paths)

    # Return the video path
    return video_path

def outpaint(image_path, mask_path, prompt, target, inference_num, amount=1, number=0):
    """
    Perform the inpainting process on the given image_path using the provided mask_path, prompt, target, inference_num, amount, and number.
    Returns a list of image_paths for the inpainted frames.
    """
    print("Processing frame " + str(number) + "...\n with prompt: " + prompt + "\nand inference_num: " + str(inference_num) + "\n")
    # Load the image and mask
    image = Image.open(image_path)
    mask_image = Image.open(mask_path)
    image_paths = []

    # Use the global model
    global global_model
    pipe = global_model
    neg_prompt = '2d, drawing, vector, paint, sketch, handdraw, weird, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad proportions, cloned face, deformed, dehydrated, disfigured, duplicate, extra arms, extra fingers, extra legs, extra limbs, fused fingers, gross proportions, long neck, malformed limbs, missing arms, missing legs, morbid, mutated hands, mutation, mutilated, out of frame, poorly drawn face, poorly drawn hands, too many fingers, ugly'
    # Run the inpainting
    output = pipe(prompt=[prompt] * amount, image=image, mask_image=mask_image, height=image.height, width=image.width, num_inference_steps=inference_num, guidance_scale=7.6, negative_prompt=[neg_prompt] * amount)

    for img in output.images:
        number += 1
        # Open the input image and resize it to match the output image size
        input_img = Image.open(image_path).resize(img.size)

        # Paste the input image onto the output image
        img.paste(input_img, (0, 0), input_img)
        print("Saving frame " + str(number) + "...\n with prompt: " + prompt + "\nand inference_num: " + str(inference_num) + "\n")
        local_image_path = "static/" + target + "/frame" + str(number) + ".png"
        img.save(local_image_path)
        image_paths.append(local_image_path)
    return image_paths

def create_mask(input_image_path):
    """
    Create a binary mask for the input_image_path, where the mask is white where the input image has an alpha channel of 0.
    Returns the temporary file path of the created mask image.
    """
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
    """
    Resize the input_image_path to half its size and paste it into the center of a new blank image with the original size.
    Returns the temporary file path of the zoomed-out image.
    """
    img = Image.open(input_image_path)
    w, h = img.size
    img = img.resize((w // 2, h // 2))  # resize the image to half its size
    img_out = Image.new("RGBA", (w, h), (255, 255, 255, 0))  # create a new blank image with the original size
    img_out.paste(img, (w // 4, h // 4))  # paste the resized image into the center of the new image

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_img:
        img_out.save(temp_img.name)
        return temp_img.name

def zoom_at(img, zoom, x, y):
    """
    Zoom the given img at the specified x and y coordinates by the provided zoom factor.
    Returns the zoomed image.
    """
    w, h = img.size

    # Expand the region to zoom out
    xz = x * zoom / 2
    yz = y * zoom / 2
    box = (int(max(0, x - xz)), int(max(0, y - yz)), int(min(w, x + xz)), int(min(h, y + yz)))
    cropped_img = img.crop(box)

    # Resize the cropped image back to the original size using OpenCV
    cropped_img_cv = np.array(cropped_img)
    # Change the interpolation method to INTER_LANCZOS4 for a sharper image
    resized_img_cv = cv2.resize(cropped_img_cv, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # Convert the resized image back to PIL image
    resized_img = Image.fromarray(resized_img_cv)

    return resized_img

def decode_base64_image(base64_img):
    """
    Decode the given base64_img string into image data.
    Returns the decoded image data.
    """
    if 'base64,' in base64_img:
        base64_img = base64_img.split('base64,')[1]
    padding = 4 - len(base64_img) % 4
    if padding:
        base64_img += "=" * padding
    img_data = base64.b64decode(base64_img)
    return img_data

def animate(frame_count, frames, fps, target, image_base64s):
    """
    Create a video animation by zooming out and in of the images provided in image_base64s.
    The video will have the specified frame_count, frames, fps, and target.
    Returns the path of the created video file.
    """
    log_scale = math.log(2)
    with imageio.get_writer('static/' + target + ".mp4", mode='I', fps=fps) as writer:
        for i in range(frame_count + 1):
            img_data = decode_base64_image(image_base64s[i])
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
            img_data = decode_base64_image(image_base64s[i])
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
    """
    Perform the zoom-out and inpainting process on the given image_path using the provided target, prompt, inference_num, amount, and number.
    Returns the list of image_paths for the inpainted frames.
    """
    # Ensure the directory exists
    os.makedirs("static/" + target, exist_ok=True)
    Image.open(image_path).save("static/" + target + "/frame" + str(number) + ".png")
    print(f"Processing frame {str(number)}...")
    image_path = zoom_out(image_path)
    mask_path = create_mask(image_path)
    image_path = outpaint(image_path, mask_path, prompt, target, inference_num, amount, number)
    return image_path

if __name__ == '__main__':
    print(init_msg)
    global_model = load_model()
    app.run(debug=False, port=5002, host='0.0.0.0')