from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import os
import shutil
from openai import OpenAI
import streamlit as st
from openai import OpenAI
from langchain.prompts import ChatPromptTemplate



CHROMA_PATH = 'chroma'
PROMPT_TEMPLATE = """
Use the following pieces of context to answer the question. If you don't know the answer, or if the answer doesn't fall within the context, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:
"""

from dotenv import load_dotenv
import os
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
OpenAI.api_key = api_key

def load_documents():
    document_loader = PyPDFDirectoryLoader('data')
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def get_embedding_function():
    embeddings = OpenAIEmbeddings()
    return embeddings

def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])  
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks):



    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    client = OpenAI()
    response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
            {"role": "system", "content": "You are a helpful assistant that answers based on the provided context."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )


    response_text = response.choices[0].message.content
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    
    return response_text

def generate_response(input_text):
    response = query_rag(input_text)
    st.info(response)


if __name__ == "__main__":
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)





    st.title("Condition Report Chatbot")


    with st.form("my_form"):
        text = st.text_area(
            # "Enter text:",
            "Ask me anything about the 2016 CINMS CR:",
        )
        submitted = st.form_submit_button("Submit")
        # if not openai_api_key.startswith("sk-"):
        #     st.warning("Please enter your OpenAI API key!", icon="⚠")
        if submitted: #and openai_api_key.startswith("sk-"):
            generate_response(text)

