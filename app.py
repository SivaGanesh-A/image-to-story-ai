from flask import Flask, render_template, request, redirect, url_for, session
import os
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
from PIL import Image
from werkzeug.utils import secure_filename

# Flask App Initialization
app = Flask(__name__)
app.secret_key = "your_secret_key"  # Required for sessions
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Generate Caption using BLIP
def generate_caption(image_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, return_tensors="pt")
    caption_ids = model.generate(**inputs)
    caption = processor.decode(caption_ids[0], skip_special_tokens=True)
    return caption

# Generate Story using GPT
def generate_story(caption):
    try:
        # Use GPT-2 to generate the story
        story_generator = pipeline("text-generation", model="gpt2")
        story = story_generator(caption, max_length=300, num_return_sequences=1)
        return story[0]["generated_text"]
    except Exception as e:
        return f"An error occurred: {e}"

@app.route("/", methods=["GET", "POST"])
def about():
    return render_template("about.html")

@app.route("/home", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Save the uploaded image
        image = request.files['image']
        filename = secure_filename(image.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(image_path)

        # Generate caption and story
        caption = generate_caption(image_path)
        story = generate_story(caption)

        # Store results in the session
        session['image_path'] = image_path
        session['caption'] = caption
        session['story'] = story

        # Redirect to the result page
        return redirect(url_for('generate_story_page'))
    return render_template("index.html")

@app.route("/generate-story")
def generate_story_page():
    # Fetch data from session
    image_path = session.get('image_path')
    caption = session.get('caption')
    story = session.get('story')

    # Render the result page
    return render_template("story_result.html", image_path=image_path, caption=caption, story=story)

if __name__ == "__main__":
    app.run(debug=False)
