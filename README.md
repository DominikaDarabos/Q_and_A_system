# Q_and_A_system

## Simple RAG System Implementation
This repository contains a simple implementation of a Retrieval-Augmented Generation (RAG) - like system, showing a pipeline using easily accessible and open-source tools and resources.

### Workflow
1. DATA PREPARATION: 
- Upload text files to the designated folder for preprocessing.
or
- Use the upload_directory.py script to generate text files in the specified folder from Hugging Face's Wikipedia dataset.

2. RETRIEVAL:
* Chunking, Embedding, Indexing: 
    - Chunks can be created as:
        + non-overlapping
        + overlapping
        + context-aware (splitting at easily recognizable sentence boundaries such as [? ! . , -])
    - The pipeline creates embeddings from the chunks and converts them into an index.
    - The generated index is saved in an index.pkl file. If this file exists and the REGENERATE_INDEX variable is set to False, the system loads it instead of regenerating it.
* Question processing:
    - The incoming question is converted to an embedding.
    - The system performs a similarity search in the index and selects the K nearest neighbors (KNN) to provide context for text generation by the LLM.

3. GENERATION:
Based on the retrieved context, the pre-trained LLM generates an answer. This is the most time-consuming step in the pipeline.


### Pre-trained models:
* The Embedder class uses the BERT "bert-base-uncased" model for embeddings.
* The Indexer class uses FAISS for efficient similarity search.
* Text generation is powered by models from the LLMWare library, which provides around 50 models for use in its framework.
    - The default text generation model is onnxruntime_genai.
    - Other models are available but may require more powerful hardware.
    - Even with onnxruntime_genai, running on a standard laptop or Google Colab can take time, even for simpler queries.

### Global variables

DATA_DIRECTORY = The directory from which the model processes text files for information retrieval.
LLM_MODEL_NAME = The name of the chosen model from the LLMWare library.
NUM_MODEL_MAX_OUTPUT_TOKEN = The maximum number of tokens the model can use for text generation. Higher values allow longer answers but increase computational cost.
K_KNN_CONTENT = The number of chunks selected for similarity-based retrieval to provide context.
REGENERATE_INDEX = Booleand to show if the saved index should be regenerated or not. It is useful when additional documents are uploaded to the data directory.


### Dependencies
Required Libraries:
transformers (for BERT embeddings)
faiss (for indexing)
llmware (for text generation)
onnxruntime (for the default LLM model through llmware)
Additional standard libraries (os, pickle, etc.)

To install the dependencies, use the following command:

```pip install transformers faiss-gpu datasets llmware onnxruntime-genai```


### MODEL ACCURACY MEASUREMENT
Depends on the context in which it will be used. If quicker responses are prioritized over absolute accuracy, then response time should also be a critical metric. For applications dealing with highly complex topics, accuracy must be on focus.
* Test the context retrieval:
    - Check the quality of the retrieved chunks in relation to the input query.
    - Classify the retrieved contexts as relevant or not relevant. Then, precision and recall metrics can be applied to the retrieved context.
* Text generation:
    - Use a collection of predefined input-output pairs to evaluate the generated text against expected answers.
    - Human Feedback: Assess the model output based on:
        + correctness
        + verbosity
        + relevance
        + helpfulness
        + harmfulness
        + verbosity

* Use explainable AI algorithms such as Integrated Gradients to analyze model's decision-making process. These algorithms assign relevance scores to tokens or words in the input query to identify which components had the most impact on the output. 


### PROVIDING THE SOLUTION TO THE END USER
Depending on the context to which the system should be used, some potential outcomes:
1. Users could interact with the system through a simple local application with an input field for the question and an output field to display the answer. The application could connect to a cloud-based database for retrieving additional or updated information.
Challenges: Running the system locally may require powerful hardware resources, which could impact efficiency, especially for complex queries.
2. Deploy the system as a web application accessible via a browser. However, in that case, multi-threading or concurrency handling must be implemented.
2. Integrate the RAG system as a chatbot or FAQ module in existing platforms for addressing frequently asked questions or support systems.


### Example run:

```python RAG.py```

Output:

Index and text chunks loaded.
Index built or loaded in: 0.0 seconds

Enter your question: When did Abraham Lincoln live?


__Question: When did Abraham Lincoln live?__
__Answer: Abraham Lincoln lived from February 12, 1809, to April 15, 1865.__

Text generated in: 23.7 seconds
