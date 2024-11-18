from llmware.models import ModelCatalog
import os, time, re
import tqdm
import faiss
import pickle
import numpy as np
from importlib import util

from transformers import BertModel, BertTokenizer
import torch

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Chunker:
    def __init__(self, chunk_size=512, step_size=256):
        self.chunk_size = chunk_size
        self.step_size = step_size

    def get_chunks(self, data):
        for text in data:
            for i in range(0, len(text), self.step_size):
                max_size = min(self.chunk_size, len(text) - i)
                yield text[i:i+max_size]

class ContextAwareChunker():
    def __init__(self, chunk_size=512):
        self.chunk_size = chunk_size

    def get_chunks(self, data):
        for text in data:
            sentences = re.split(r'(?!.)\s+', text)

            chunk = ""
            for sentence in sentences:
                if len(chunk) + len(sentence) > self.chunk_size:
                    yield chunk
                    chunk = sentence
                else:
                    chunk += " " + sentence
            if chunk:
                yield chunk


class TextEmbedder:
    def __init__(self, model_name="bert-base-uncased"):
        # Load the pre-trained BERT model and tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()

    def embed(self, text):
        # Tokenize the input text and convert it to tensor format
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        # Get the output embeddings from BERT
        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state[:, 0, :].squeeze()


class DirectoryLoader:
    def __init__(self, directory, batch_size=512, chunker=ContextAwareChunker()):
        self.directory = os.path.join(os.getcwd(), directory)
        self.chunker = chunker
        self.batch_size = batch_size

    def load(self):
        for root, dirs, files in os.walk(self.directory):
            for file in files:
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    yield f.read()

    def get_chunks(self):
        return self.chunker.get_chunks(self.load())

    def get_chunk_batches(self):
        chunks = []
        for chunk in self.get_chunks():
            chunks.append(chunk)
            if len(chunks) >= self.batch_size:
                yield chunks
                chunks = []

        if len(chunks) > 0:
            yield chunks

    def __iter__(self):
        return self.get_chunk_batches()

class Indexer:
    def __init__(self):
        self.loader = DirectoryLoader("data2")
        self.index = None
        self.index_file = "index.pkl"
        self.embedder = TextEmbedder()
        if self._index_exists():
            self.load_index()
        else:
            self.build_index()
            self.save_index()

    def _index_exists(self):
        """Check if the index file already exists."""
        try:
            with open(self.index_file, 'rb') as f:
                return True
        except FileNotFoundError:
            return False

    def build_index(self):
        self.content_chunks = []
        self.index = None
        #for chunk_batch in tqdm(self.loader):
        for chunk_batch in self.loader:
            embeddings = self.get_embeddings(chunk_batch)
            if self.index is None:
                self.index = faiss.IndexFlatL2(len(embeddings[0]))
            self.index.add(embeddings)
            self.content_chunks.extend(chunk_batch)

    def save_index(self):
        """Save the built index and content chunks to a file."""
        with open(self.index_file, 'wb') as f:
            pickle.dump((self.index, self.content_chunks), f)
        print("Index and chunks saved.")

    def load_index(self):
        """Load the index and content chunks from the file."""
        with open(self.index_file, 'rb') as f:
            self.index, self.content_chunks = pickle.load(f)
        print("Index and chunks loaded.")

    def get_embeddings(self, text):
        embedding = self.embedder.embed(text)
        if embedding.dim() == 0:
            embedding = embedding.unsqueeze(0)
        return embedding

    def query(self, query, k=5):
        embedding = self.get_embeddings([query])
        embedding_array = np.array(embedding).reshape(1, -1)
        _, indices = self.index.search(embedding_array, k)
        return [self.content_chunks[i] for i in indices[0]]

class QueryEngine:
    def __init__(self, index, k=3):
        self.index = index
        self.k = k
        self.model = ModelCatalog().load_model("phi-3-onnx", max_output=500)

    def preprocess_query(self, query):
        return query.lower().strip()

    def answer_question(self, question):
        query = self.preprocess_query(question)
        most_similar = self.index.query(query, k=self.k)
        prompt = "\n".join(reversed(most_similar)) + "\n\n" + question
        tokens = []
        streamed_response = self.model.stream(prompt)
        for streamed_token in streamed_response:
            tokens.append(streamed_token)
        text_out = ''.join(tokens)
        return text_out


class RetrievalAugmentedRunner:
    def __init__(self, k=5):
        self.k = k

    def train(self):
        self.index = Indexer()

    def __call__(self, query):
        query_engine = QueryEngine(self.index, k=self.k)
        return query_engine.answer_question(query)

if __name__ == "__main__":

    model = RetrievalAugmentedRunner()
    start = time.time()
    model.train()
    print("Index built in:", round(time.time() - start, 2), "seconds")

    #while True:
    question = input("\nEnter your question: ")
    # if not question.strip():
    #     print("Exiting...")
    #     break
    answer = model(question)
    text = str(answer).split("<human>")[0].strip()
    print(f"\nQuestion: {question}\nAnswer: {text}\n")
