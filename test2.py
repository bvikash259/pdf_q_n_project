import streamlit as st
import openai
import pinecone
import PyPDF2
from sentence_transformers import SentenceTransformer
import os


# Initialize OpenAI API
openai.api_key = "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"

from pinecone import  ServerlessSpec

os.environ['PINECONE_API_KEY'] ="pcsk_6Py8DZ_Y24yWBWmGKDA37i5zBF6rNiJCmeb7Xyr9VXdkFJoppQXXjSZ1uwTLKgw1owUHL"
pine_key=os.environ['PINECONE_API_KEY'] 
# Initialize Pinecone
pc=pinecone.Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Check if the index exists, then create it if not
if "my-pdf-project1" not in pc.list_indexes().names():
    pc.create_index(
        name="my-pdf-project1",
        dimension=384,  # Make sure this matches your embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Initialize Pinecone
#pinecone.init(api_key="pcsk_6Py8DZ_Y24yWBWmGKDA37i5zBF6rNiJCmeb7Xyr9VXdkFJoppQXXjSZ1uwTLKgw1owUHL")
#pinecone.create_index(
 #   name="my-new-pdf-project",
  #  dimension=1536,
   # metric="cosine"
#)
index = pc.Index("my-pdf-project1")
# Initialize Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    return text

def store_embeddings(text):
    sentences = text.split(". ")
    embeddings = model.encode(sentences).tolist()
    for i, sentence in enumerate(sentences):
        index.upsert([(f"id-{i}", embeddings[i], {"text": sentence})])

def query_pinecone(query):
    query_embedding = model.encode([query]).tolist()[0]
    results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
    return "\n".join([match["metadata"]["text"] for match in results["matches"]])

def ask_openai(question, context):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI assistant helping with document queries."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
        ]
    )
    return response["choices"][0]["message"]["content"]

# Streamlit UI

st.set_page_config(page_title="PDF Q&A with AI", layout="wide")
st.image("https://deepai.org/gallery-item/2bee6bc4cf944cdd951e86d584628f5b/create-a-background-image-for-my-website-that_fjtDL3m.jpg.html", width=200)
st.title("📄 PDF Q&A Chatbot")
st.markdown("Upload a PDF and ask any question related to its content!")

pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])
if pdf_file:
    with st.spinner("Uploading pdf.... Please wait... "):
        text = extract_text_from_pdf(pdf_file)
        store_embeddings(text)
        st.success("PDF processed and stored successfully!")

    question = st.text_input("Ask a question about the document:")
    st.spinner("searching for your answer.. please wait...")
    if question:
        
            context = query_pinecone(question)
            answer = ask_openai(question, context)
            st.write(f"### Your Answer is... :")
            st.write(answer)
