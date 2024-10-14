from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np

def insert_random_text(image_path, sentence, output_path):
    # Open the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Load a font
    try:
        font = ImageFont.truetype("arial.ttf", 40)  # You can adjust font size here
    except IOError:
        font = ImageFont.load_default()  # Fallback if font not found

    # Get the image dimensions
    img_width, img_height = image.size

    # Generate random coordinates for text placement
    x = random.randint(0, img_width)
    y = random.randint(0, img_height)

    # Get the size of the text box
    text_width, text_height = draw.textsize(sentence, font=font)

    # Ensure the text doesn't overflow the image boundaries
    x = min(x, img_width - text_width)
    y = min(y, img_height - text_height)

    # Create a blank image for rotation
    text_img = Image.new('RGBA', (text_width, text_height), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_img)
    
    # Add the text to the blank image
    text_draw.text((0, 0), sentence, font=font, fill=(255, 0, 0, 255))  # Red text
    
    # Rotate the text image randomly
    angle = random.randint(0, 360)
    rotated_text_img = text_img.rotate(angle, expand=1)

    # Get rotated text dimensions
    rotated_width, rotated_height = rotated_text_img.size

    # Adjust random position if rotation causes text to go out of bounds
    x = min(x, img_width - rotated_width)
    y = min(y, img_height - rotated_height)

    # Paste the rotated text image onto the original image
    image.paste(rotated_text_img, (x, y), rotated_text_img)

    # Save the output image
    image.save(output_path)

# Example usage
image_path = "generators/demo.jpeg"  # Path to your input image
sentence = "Happy birthday Junyi and Guanzheng"  # The sentence to insert
output_path = "generators/output_image.jpg"  # Path to save the output image

insert_random_text(image_path, sentence, output_path)
