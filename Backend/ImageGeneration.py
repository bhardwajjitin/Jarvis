# import asyncio
# from random import randint
# from PIL import Image
# import requests
# from dotenv import get_key 
# import os
# from time import sleep

# def open_images(prompt):
#     folder_path=r"Data"
#     prompt=prompt.replace(" ","_")

#     Files=[f"{prompt}_{i}.jpg" for i in range(1,5)]

#     for jpg_file in Files:
#         image_path=os.path.join(folder_path,jpg_file)

#         try:
#             img=Image.open(image_path)
#             print(f"Opening Image {image_path}")
#             img.show()
#             sleep(1)

#         except IOError:
#             print(f"Error in opening {image_path}")

# API_URL="https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
# headres={"Authorization":f"Bearer {get_key('.env','HuggingFaceAPIKey')}"}

# async def query(payload):
#     response=await asyncio.to_thread(requests.post,API_URL,headers=headres,json=payload)
#     return response.content

# async def generate_images(prompt:str):
#     tasks=[]

#     for _ in range(4):
#         payload={"inputs":f"{prompt},quality=4K,sharpness=maximum,Ultra High Details,high resolution"}
#         tasks.asyncio.create_task(query(payload))
#         tasks.append(tasks)


#     image_byte_list=await asyncio.gather(*tasks)

#     for i,image_bytes in enumerate(image_byte_list):
#         with open(fr"Data/{prompt.replace(' ',' ')}{i+1}.jpg","wb") as f:
#             f.write(image_bytes)


# def GenerateImages(prompt:str):
#     asyncio.run(generate_images(prompt))
#     open_images(prompt)


# while True:
#     try:

#         with open(r"Frontend\Files\ImageGeneration.data","r") as f:
#             Data: str=f.read()

#         Prompt, Status=Data.split(",")

#         if Status=="True":
#             print("Generating Images....")
#             ImageStatus=GenerateImages(prompt=Prompt)

#             with open(r"Frontend\Files\ImageGeneration.data","w") as f:
#                 f.write("False,False")
#                 break

#         else:
#             sleep(1)

#     except:
#         pass


import asyncio
from pathlib import Path
from PIL import Image
import requests
from dotenv import dotenv_values
import os
import time
from typing import List, Tuple
from rich.progress import Progress

# Configuration
CONFIG = {
    "API_URL": "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0",
    "DATA_DIR": "Data",
    "STATUS_FILE": "Frontend/Files/ImageGeneration.data",
    "MAX_RETRIES": 3,
    "DELAY_BETWEEN_REQUESTS": 1
}

# Load environment variables once
env_vars = dotenv_values(".env")
HEADERS = {"Authorization": f"Bearer {env_vars.get('HuggingFaceAPIKey')}"}

def ensure_data_dir() -> None:
    """Ensure the data directory exists"""
    Path(CONFIG["DATA_DIR"]).mkdir(parents=True, exist_ok=True)

async def query_api(payload: dict) -> bytes:
    """Make API request with retries and error handling"""
    for attempt in range(CONFIG["MAX_RETRIES"]):
        try:
            response = await asyncio.to_thread(
                requests.post,
                CONFIG["API_URL"],
                headers=HEADERS,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.content
        except (requests.RequestException, TimeoutError) as e:
            if attempt == CONFIG["MAX_RETRIES"] - 1:
                raise
            await asyncio.sleep(CONFIG["DELAY_BETWEEN_REQUESTS"] * (attempt + 1))
    return b''

def enhance_prompt(prompt: str) -> str:
    """Enhance the prompt with quality modifiers"""
    enhancements = [
        "4K resolution",
        "ultra detailed",
        "sharp focus",
        "professional photography",
        "high quality"
    ]
    return f"{prompt}, {', '.join(enhancements)}"

async def generate_single_image(prompt: str, index: int) -> str:
    """Generate and save a single image"""
    payload = {
        "inputs": enhance_prompt(prompt),
        "options": {"wait_for_model": True}
    }
    
    image_bytes = await query_api(payload)
    if not image_bytes:
        return ""
    
    filename = f"{prompt.replace(' ', '_')}_{index+1}.jpg"
    filepath = os.path.join(CONFIG["DATA_DIR"], filename)
    
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    
    return filepath

async def generate_images(prompt: str) -> List[str]:
    """Generate multiple images concurrently"""
    tasks = [generate_single_image(prompt, i) for i in range(3)]
    return await asyncio.gather(*tasks)

def display_images(image_paths: List[str]) -> None:
    """Display generated images with error handling"""
    for path in image_paths:
        if not path:  # Skip failed generations
            continue
            
        try:
            img = Image.open(path)
            img.show()
            time.sleep(1)  # Prevent rapid image stacking
        except Exception as e:
            print(f"[red]Error displaying {path}: {e}[/red]")

def read_status() -> Tuple[str, bool]:
    """Read and parse the status file"""
    try:
        with open(CONFIG["STATUS_FILE"], "r") as f:
            content = f.read().strip()
            if "," in content:
                prompt, status = content.split(",", 1)
                return prompt, status.lower() == "true"
    except FileNotFoundError:
        pass
    return "", False

def write_status(prompt: str = "", status: bool = False) -> None:
    """Update the status file"""
    with open(CONFIG["STATUS_FILE"], "w") as f:
        f.write(f"{prompt},{status}")

async def main_loop():
    """Main processing loop - now takes live input"""
    ensure_data_dir()

    print("[🧠 JARVIS]: Ready to generate images. Type your prompt (or 'exit' to quit):")
    while True:
        prompt = input("You: ")
        if prompt.lower() in ("exit", "quit"):
            print("[🧠 JARVIS]: Shutting down...")
            break

        try:
            print(f"[🧠 JARVIS]: Generating images for: {prompt}")
            image_paths = await generate_images(prompt)
            display_images(image_paths)
            print("[✅ Done]")
        except Exception as e:
            print(f"[❌ Error]: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\n[yellow]Shutting down image generator...[/yellow]")