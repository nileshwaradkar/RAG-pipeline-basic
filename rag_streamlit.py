import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate

st.title("Carbon Tax Query Assistant")

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=":material/person:" if msg["role"] == "user" else "ai"):
        st.markdown(msg["content"])

if question := st.chat_input("Enter your query:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(question)

    loader = PyPDFLoader("https://www.worldbank.org/content/dam/Worldbank/document/Climate/background-note_carbon-tax.pdf")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="phi3:latest")
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(documents=all_splits)

    model = ChatOllama(model="phi3:latest")
    prompt = ChatPromptTemplate.from_template(
        """
            You are a knowledgeable assistant specializing in question-answering. Please utilize the provided context to formulate your response. If the answer is not available, simply state that you do not know. Limit your response to a maximum of three concise sentences.

                Question: {question}

                Context: {context}

                Answer:
            """
    )

    retrieved_docs = vector_store.similarity_search(question)
    answers = "\n\n".join(doc.page_content for doc in retrieved_docs)
    formatted_prompt = prompt.format(question=question, context=answers)
    answer = model.invoke(formatted_prompt)

    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer.content})
    with st.chat_message("assistant"):
        st.markdown(answer.content)
