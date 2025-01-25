from transformers import pipeline

# Load the text generation pipeline
story_generator = pipeline("text-generation", model="gpt2")

# Test input
caption = "Once upon a time in a distant galaxy"
story = story_generator(caption, max_length=150, num_return_sequences=1)

# Print the generated story
print(story[0]["generated_text"])
