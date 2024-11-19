from datasets import load_dataset
import os, re

NUMBER_OF_FILES = 10
OUTPUT_DIR = "rag_data"

dataset = load_dataset("wikipedia", "20220301.en", trust_remote_code=True)
dataset = dataset['train'][:NUMBER_OF_FILES]

def write_wikipedia_samples_from_dataset(dataset, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    titles = dataset['title']
    texts = dataset['text']
    for i in range(NUMBER_OF_FILES):
        title = titles[i]
        text = texts[i]
        clean_title = re.sub(r'[\\/:*?"<>|]', '_', title)
        filename = os.path.join(output_dir, f"{i + 1}_{clean_title}.txt")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Title: {title}\n\n{text}")

write_wikipedia_samples_from_dataset(dataset=dataset, output_dir=OUTPUT_DIR)