import faiss
import time
import numpy as np
from tqdm import tqdm
import lamini
import re, os
import pickle

lamini.api_key = "<API KEY>"

class DefaultChunker:
    def __init__(self, chunk_size=512, step_size=256):
        self.chunk_size = chunk_size
        self.step_size = step_size

    def get_chunks(self, data):
        for text in data:
            for i in range(0, len(text), self.step_size):
                max_size = min(self.chunk_size, len(text) - i)
                yield text[i:i+max_size]

class DirectoryLoader:
    def __init__(self, directory, batch_size=512, chunker=DefaultChunker()):
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
            if len(chunks) == self.batch_size:
                yield chunks
                chunks = []

        if len(chunks) > 0:
            yield chunks

    def __iter__(self):
        return self.get_chunk_batches()

class LaminiIndex:
    def __init__(self, loader):
        self.loader = loader
        self.index = None
        self.index_file = "index.pkl" 
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
        for chunk_batch in tqdm(self.loader):
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

    def get_embeddings(self, examples):
        ebd = lamini.api.embedding.Embedding()
        embeddings = ebd.generate(examples)
        result = np.array([embedding[0] for embedding in embeddings])
        return result


    def query(self, query, k=5):
        embedding = self.get_embeddings([query])[0]
        embedding_array = np.array([embedding])
        _, indices = self.index.search(embedding_array, k)
        return [self.content_chunks[i] for i in indices[0]]

class QueryEngine:
    def __init__(self, index, k=5):
        self.index = index
        self.k = k
        self.model = lamini.Lamini(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct")

    def preprocess_query(self, query):
        return query.lower().strip()

    def answer_question(self, question):
        query = self.preprocess_query(question)
        most_similar = self.index.query(query, k=self.k)
        prompt = "\n".join(reversed(most_similar)) + "\n\n" + question
        return self.model.generate(f"<s>[INST]{prompt}[/INST]")

class RetrievalAugmentedRunner:
    def __init__(self, dir, k=5):
        self.k = k
        self.loader = DirectoryLoader(dir)

    def train(self):
        self.index = LaminiIndex(self.loader)

    def __call__(self, query):
        query_engine = QueryEngine(self.index, k=self.k)
        return query_engine.answer_question(query)

def main():
    model = RetrievalAugmentedRunner(dir="data")
    start = time.time()
    model.train()
    print("Index built in:", round(time.time() - start, 2), "seconds")

    while True:
        question = input("\nEnter your question: ")
        if not question.strip():
            print("Exiting...")
            break
        answer = model(question)
        text = str(answer)
        clean_text = re.sub(r'\[INST\]s\[/INST\]', '', text)
        print(f"\nQuestion: {question}\nAnswer: {clean_text}\n")

main()