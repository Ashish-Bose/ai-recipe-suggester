from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import torch
import gradio as gr

# Load dataset
dataset = load_dataset("muhammadomair678/recipe-dataset", split="train[:1000]")
recipe_ingredients = [example['prompt'] for example in dataset]
embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
recipe_embeddings = embedder.encode(recipe_ingredients, convert_to_tensor=True, show_progress_bar=True, batch_size=16)

def suggest_recipes(user_ingredients, top_k=3):
    user_embedding = embedder.encode(user_ingredients, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(user_embedding, recipe_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)
    suggestions = []
    for score, idx in zip(top_results[0], top_results[1]):
        recipe = dataset[int(idx)]
        suggestions.append({
            'completion': recipe['completion'],
            'score': score.item()
        })
    return suggestions

def gradio_suggest(user_input):
    if not user_input.strip():
        return "Please enter some ingredients!"
    suggestions = suggest_recipes(user_input)
    output = ""
    for i, sug in enumerate(suggestions, 1):
        output += f"**Suggestion {i}** (Match Score: {sug['score']:.2f})\n\n"
        output += f"{sug['completion']}\n\n---\n\n"
    return output

iface = gr.Interface(
    fn=gradio_suggest,
    inputs=gr.Textbox(label="Enter your pantry ingredients (e.g., chicken, rice, tomatoes)"),
    outputs=gr.Markdown(label="Recipe Suggestions"),
    title="Pantry-to-Plate AI: Smart Recipe Generator",
    description="Input ingredients and get full recipe ideas!"
)

# Streamlit runs the app directly (no share=True)
iface.launch()
