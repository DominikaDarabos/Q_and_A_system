from llmware.models import ModelCatalog
import os, time, re
import tqdm
import faiss
import pickle
import numpy as np
from importlib import util

from transformers import BertModel, BertTokenizer
import torch

DATA_DIRECTORY = "rag_data"
LLM_MODEL_NAME = "phi-3-onnx"
NUM_MODEL_MAX_OUTPUT_TOKEN = 100
K_KNN_CONTENT = 3
REGENERATE_INDEX = False

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Chunker:
    """
    Splits text into smaller chunks for indexing.

    Attributes:
        chunk_size (int): Maximum size of a chunk.
        step_size (int): Step size for creating overlapping chunks.
    Methods:
        get_chunks(data): Return a generator that produces text chunks based on fixed size from the provided data.
    """
    def __init__(self, chunk_size: int = 512, step_size: int = 256):
        self.chunk_size = chunk_size
        self.step_size = step_size

    def get_chunks(self, data):
        for text in data:
            for i in range(0, len(text), self.step_size):
                max_size = min(self.chunk_size, len(text) - i)
                yield text[i:i+max_size]

class ContextAwareChunker():
    """
    Splits text into context-aware chunks based on sentence boundaries.

    Attributes:
        chunk_size (int): Maximum size of a chunk.
    Methods:
        get_chunks(data): Return a generator that produces text chunks based on sentence boundaries from the provided data.
    
    """
    def __init__(self, chunk_size: int = 512):
        self.chunk_size = chunk_size

    def get_chunks(self, data):
        for text in data:
            sentences = re.split(r'([?!.,-])', text)

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
    """
    Uses a pre-trained BERT model to embed text into vector representations.

    Attributes:
        tokenizer: BERT tokenizer for preprocessing input text.
        model: BERT model for generating embeddings.
    Methods:
        embed(text): Returns the embedding for a given input text.
    """
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()

    def create_embedding(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state[:, 0, :].squeeze()


class DataLoader:
    """
    Loads text files from a directory, chunks them, and organizes them into batches.

    Attributes:
        directory (str): Directory containing text files.
        chunker: A chunker object.
        batch_size (int): Size of each batch of chunks.
    Methods:
        load(): Yields the contents of all text files.
        get_chunks(): Yields chunks of text generated by the chunker.
        get_chunk_batches(): Groups chunks into batches for processing, ensuring each batch's size is at least as the specified batch_size.
        __iter__(): Allows iteration over chunk batches.
    """
    def __init__(self, directory: str, batch_size: int = 512, chunker = Chunker()):
        self.directory = directory
        self.chunker = chunker
        self.batch_size = batch_size

    def load(self):
        for root, _, files in os.walk(self.directory):
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
    """
    Handles embedding text and building/searching an FAISS index for retrieval.

    Attributes:
        loader: A DirectoryLoader instance for loading and chunking text.
        index (FAISS Index): Index for storing and querying embeddings.
        index_file (str): Filename for saving/loading the index.
        embedder: A TextEmbedder instance for embedding text.
        content_chunks (list): List of all indexed text chunks.
    Methods:
        _index_exists(): Checks if the index file exists.
        build_index(): Builds the FAISS index from text chunks.
        save_index(): Saves the FAISS index and chunks to a file.
        load_index(): Loads the FAISS index and chunks from a file.
        get_embeddings(text): Returns embeddings for a given text.
        query(query, k): Finds the k most similar chunks to the query based on KNN.
    """
    def __init__(self):
        self.loader = DataLoader("data2")
        self.index = None
        self.index_file = "index.pkl"
        self.embedder = TextEmbedder()
        if self._index_exists() and not REGENERATE_INDEX:
            self.load_index()
        else:
            self.build_index()
            self.save_index()

    def _index_exists(self):
        try:
            with open(self.index_file, 'rb') as f:
                return True
        except FileNotFoundError:
            return False

    def build_index(self):
        print("Start building index...")
        self.content_chunks = []
        self.index = None
        for chunk_batch in self.loader:
            embeddings = self.get_embeddings(chunk_batch)
            if self.index is None:
                self.index = faiss.IndexFlatL2(len(embeddings[0]))
            self.index.add(embeddings)
            self.content_chunks.extend(chunk_batch)

    def save_index(self):
        with open(self.index_file, 'wb') as f:
            pickle.dump((self.index, self.content_chunks), f)
        print("Index and text chunks saved.")

    def load_index(self):
        with open(self.index_file, 'rb') as f:
            self.index, self.content_chunks = pickle.load(f)
        print("Index and text chunks loaded.")

    def get_embeddings(self, text: str):
        embedding = self.embedder.create_embedding(text)
        if embedding.dim() == 0:
            embedding = embedding.unsqueeze(0)
        return embedding

    def query(self, question: str, k: int = 5):
        embedding = self.get_embeddings([question])
        embedding_array = np.array(embedding).reshape(1, -1)
        _, indices = self.index.search(embedding_array, k)
        return [self.content_chunks[i] for i in indices[0]]

class RAGengine:
    """
    Handles question answering by combining retrieval and generation.

    Attributes:
        index: An Indexer instance for retrieving relevant context.
        k (int): Number of similar chunks to retrieve.
        model: A text generation model from llmware ModelCatalog.
    Methods:
        preprocess_query(query): Prepares the query by trimming.
        answer_question(question): Retrieves context, generates an answer, and returns it.
        __call__(query): Shortcut for calling answer_question.
    """
    def __init__(self):
        self.index = Indexer()
        self.model = ModelCatalog().load_model(LLM_MODEL_NAME, max_output=NUM_MODEL_MAX_OUTPUT_TOKEN)
        #self.model = ModelCatalog().load_model("llmware/bling-tiny-llama-v0", max_output=500)
        #self.model = ModelCatalog().load_model("bartowski/Meta-Llama-3-8B-Instruct-GGUF", max_output=500)

    def preprocess_query(self, query):
        return query.strip()

    def answer_question(self, question):
        query = self.preprocess_query(question)
        most_similar = self.index.query(query, k=K_KNN_CONTENT)
        prompt = "\n".join(reversed(most_similar)) + "\n\n" + question
        tokens = []
        streamed_response = self.model.stream(prompt)
        for streamed_token in streamed_response:
            tokens.append(streamed_token)
        text_out = ''.join(tokens)
        return text_out
    
    def __call__(self, query):
        return self.answer_question(query)

if __name__ == "__main__":

    model = RAGengine()
    start = time.time()
    print("Index built or loaded in:", round(time.time() - start, 2), "seconds")

    question = input("\nEnter your question: ")
    start = time.time()
    answer = model(question)
    text = str(answer).split("<human>")[0].strip()
    print(f"\nQuestion: {question}\nAnswer: {text}\n")
    print("Text generated in:", round(time.time() - start, 2), "seconds")
