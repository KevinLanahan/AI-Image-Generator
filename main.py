import tkinter as tk
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image, ImageTk

print("Loading model...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    safety_checker=None
)
pipe = pipe.to("cpu")
print("Model loaded.")


#Tkinter window
root = tk.Tk()
root.geometry("800x600")
root.title("AI Image Generator")

prompt_label = tk.Label(
    root,
    text="Enter a prompt to generate an image:",
    font=('Comic Sans MS', 16, 'normal'),
    padx=20,
    pady=20
)
prompt_label.grid(row=0, column=0, columnspan=2)

prompt_input = tk.Entry(
    root,
    font=('Comic Sans MS', 16, 'normal'),
    width=40
)
prompt_input.grid(row=1, column=0, columnspan=2, padx=20)

displayed_prompt_label = tk.Label(
    root,
    text="",
    font=('Comic Sans MS', 14, 'italic'),
    pady=10
)
displayed_prompt_label.grid(row=3, column=0, columnspan=2)

image_label = tk.Label(root)
image_label.grid(row=4, column=0, columnspan=2, pady=20)

def generate_image():
    user_prompt = prompt_input.get()
    if not user_prompt:
        displayed_prompt_label.config(text="Please enter a prompt first!")
        return
    
    print(f"Generating image for prompt: '{user_prompt}'")

    image = pipe(user_prompt).images[0]

    output_path = "outputs/generated_image.png"
    image.save(output_path)

    display_image = image.resize((512, 512))
    tk_image = ImageTk.PhotoImage(display_image)

    image_label.config(image=tk_image)
    image_label.image = tk_image  

    displayed_prompt_label.config(text=f"Prompt: {user_prompt}")

generate_button = tk.Button(
    root,
    text="Generate Image",
    font=('Comic Sans MS', 14, 'bold'),
    command=generate_image,
    padx=10,
    pady=10
)
generate_button.grid(row=2, column=0, columnspan=2, pady=10)

root.mainloop()
