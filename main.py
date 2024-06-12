import streamlit as st
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import dotenv
from langchain_core.messages import HumanMessage, AIMessage


# Load environment variables from .env file
dotenv.load_dotenv()

# Initialize the language model
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Function to load and split documents
def load_and_split_document(url):
    loader = WebBaseLoader(url)
    print(loader)
    docs = loader.load()
    # print(docs)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(docs)
    return splits

# Function to initialize vectorstore
def initialize_vectorstore(doc_splits):
    vectorstore = Chroma.from_documents(documents=doc_splits, embedding=OpenAIEmbeddings())
    return vectorstore

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Initialize Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "url" not in st.session_state:
    st.session_state.url = ""

#Initial UI
st.set_page_config(page_title="RAG Bot")
st.title("OpenAI RAG Model")
st.markdown("---")

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)

# URL input
url_input = st.text_input("Enter a URL to scrape and use for context:", value=st.session_state.url)
if url_input:
    st.session_state.url = url_input

# Add queries to chat history
user_query = st.chat_input("Ask something about your URL")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        # Load and split documents  
        with st.spinner("Thinking"):
            if st.session_state.url:
                doc_splits = load_and_split_document(st.session_state.url)
                vectorstore = initialize_vectorstore(doc_splits)
                retriever = vectorstore.as_retriever()
                #pull a prompt template from LangChain's Library
                prompt = hub.pull("rlm/rag-prompt")
                #Input user_query as the question, context from the retriever (DB). Prompt into LLM, and parse output. 
                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                response = st.write_stream(rag_chain.stream(user_query))
                st.session_state.chat_history.append(AIMessage(response))
            else:
                st.error("Please enter a valid URL.")
