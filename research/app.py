import streamlit as st
import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory  # Import memory class
from dotenv import load_dotenv

load_dotenv()
groq = os.environ.get("groq")

st.title("Question Answering Chatbot")

url = st.text_input("Enter a URL:", "")

# Initialize memory
memory = ConversationBufferMemory()

if url:
    embedding = HuggingFaceEmbeddings()
    llm = ChatGroq(temperature=0, api_key=groq, model_name="llama-3.3-70b-versatile")

    loader = UnstructuredURLLoader(urls=[url])
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=150, separators=["\n\n", "\n", ".", " "]
    )
    splits = text_splitter.split_documents(loader.load())

    db = FAISS.from_documents(splits, embedding)

    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer and don't find it in the given context, just say that you don't know, don't try to make up an answer.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    query = st.text_input("Enter your question:", "")

    if query:
        # Store the query in memory
        memory.add_user_message(query)

        result = qa_chain({"query": query})
        st.write(result["result"])

        # Store the response in memory
        memory.add_ai_message(result["result"])

        st.write("Source documents:")
        for doc in result["source_documents"]:
            st.write(doc.page_content)

        # Display conversation history
        st.write("Conversation History:")
        st.write(memory.get_history())
