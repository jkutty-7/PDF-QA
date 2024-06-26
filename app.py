import streamlit as st
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import openai
from dotenv import load_dotenv, find_dotenv
import os
import cohere


# Load environment variables
_ = load_dotenv(find_dotenv())

# Set up OpenAI API key
openai.api_key = os.environ['OPENAI_API_KEY']

co = cohere.Client(os.environ.get("COHERE_API_KEY"))


import re

def get_collection_name(filename):
  base_name = filename.split(".")[0]
  
  collection_name = re.sub(r"[^\w\-_]", "", base_name)
  
  collection_name = collection_name.lower()
  
  return collection_name



def augment_multiple_query(query, model="gpt-3.5-turbo-0125"):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert multilingual document research assistant. The users are asking questions about the document so you need to create questions which will actually related to the query. "
            "Suggest up to five additional related questions to help them find the information they need, for the provided question. "
            "Suggest only short questions without compound sentences. Suggest a variety of questions that cover different aspects of the topic."
            "Make sure they are complete questions, and that they are related to the original question."
            "Output one question per line. Do not number the questions. it should be in document language itself"
        },
        {"role": "user", "content": query}
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    content = content.split("\n")
    return content

# Set up Streamlit app
st.title("PDF Document Question Answering System")

# Sidebar
with st.sidebar:
    st.subheader("PDF Document Upload")
    uploaded_file = st.file_uploader("Upload a legal document", type="pdf")

    if uploaded_file is not None:
        # Read PDF file
        reader = PdfReader(uploaded_file)
        pdf_texts = [p.extract_text().strip() for p in reader.pages]
        pdf_texts = [text for text in pdf_texts if text]

        # Split text
        character_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=1000,
            chunk_overlap=0
        )
        character_split_texts = character_splitter.split_text('\n\n'.join(pdf_texts))

        token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
        token_split_texts = []
        for text in character_split_texts:
            token_split_texts += token_splitter.split_text(text)

        # Initialize ChromaDB
        embedding_function = SentenceTransformerEmbeddingFunction()
        chroma_client = chromadb.Client()
        # collection_name = uploaded_file.name.split(".")[0].replace(" ","") 
        collection_name = get_collection_name(uploaded_file.name)
        # Check if the collection exists, if not, create a new one
        chroma_collection = chroma_client.get_or_create_collection(collection_name, embedding_function=embedding_function)
        st.write(f"Created new collection: {collection_name}")

        ids = [str(i) for i in range(len(token_split_texts))]
        chroma_collection.add(ids=ids, documents=token_split_texts)


# Main content area
original_query = st.text_input("Enter your question about the legal document:", key="query_input")

if st.button("Get Answer"):
    if not uploaded_file:
        st.error("Please upload a legal document first.")
    else:
        # Retrieve related documents
        augmented_query = augment_multiple_query(original_query)
        joint_query = [original_query] + augmented_query 
        results = chroma_collection.query(query_texts=joint_query, n_results=10, include=['documents', 'embeddings'])
        retrieved_documents = results['documents']
        unique_documents = set()
        for documents in retrieved_documents:
            for document in documents:
                unique_documents.add(document)
        unique_documents = list(unique_documents)

        # Rerank documents
        response = co.rerank(
            model="rerank-multilingual-v3.0",
            query=original_query,
            documents=unique_documents,
            top_n=10,
            return_documents=True
        )
        key_texts = "".join([item.document.text for item in response.results])

        # Generate answer using RAG
        messages = [
        {
        "role": "system",
        "content": "You are an expert in understanding multilingual documents and providing answers. Your responses should be based solely on the information provided in the document. If the answer is not present in the given information, you should explicitly state that the answer cannot be found in the document. Do not make up or fabricate any information.Give answers in bullet points"
        },
        {
        "role": "user",
        "content": f"Question: {original_query}. \n Information: {key_texts}"
        }
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            temperature=0.4,
            messages=messages,
        )
        content = response.choices[0].message.content

        st.subheader("Answer:")
        st.markdown(content)