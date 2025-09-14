import os
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# 1. Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# 2. Setup embeddings & LLM
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=api_key
)
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key, temperature=0.7)

# 3. Load dataset
with open("ai_concepts.txt", "r") as f:
    data = f.read()

# Split text into chunks
splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_text(data)

# Create vector database
vectorstore = Chroma.from_texts(chunks, embedding=embeddings)

# Setup RAG pipeline
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True
)

# ----------------- STREAMLIT UI -----------------
st.set_page_config(page_title="AI Concept Explainer", page_icon="🤖", layout="centered")

st.title("🤖 AI Concept Explainer (RAG Powered)")
st.write("Ask about AI topics like **NLP, CV, CNN, RAG, ML, Deep Learning** and choose how you want the explanation!")

# Input fields
query = st.text_input("🔎 Enter your question (e.g., 'Explain CNN as if I’m 10 years old')")

style = st.selectbox(
    "📝 Choose explanation style",
    ["Beginner-friendly", "Kid-friendly", "Technical/Expert"]
)

if st.button("Generate Explanation"):
    if query.strip() == "":
        st.warning("Please enter a question first!")
    else:
        # Modify query with style
        final_query = f"Explain this: {query}. Style: {style}"
        with st.spinner("Generating explanation..."):
            answer = qa.run(final_query)

        # Show result
        st.success("✅ Explanation generated!")
        st.markdown(f"### 🧾 Answer:\n{answer}")