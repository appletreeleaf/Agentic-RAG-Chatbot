# Logging library
from loguru import logger

# Streamlit
import streamlit as st

# Utility functions
from utills import (print_message, get_session_history, StreamHandler)

# LangChain Core
from langchain import hub
from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

# LangChain OpenAI
from langchain_openai import ChatOpenAI

# Document loaders
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader, TextLoader

# Embeddings
from langchain.embeddings import OpenAIEmbeddings

# Vector store
from langchain.vectorstores import FAISS

# Text splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Retrievers
from langchain.retrievers import BM25Retriever, EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker

# LangChain tools
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool

# Agents
from langchain.agents import create_openai_functions_agent, AgentExecutor

# Message history
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_teddynote.messages import stream_response

# Cross encoders
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

def get_loader(file_name):
    """
    Returns the appropriate loader based on the document format.

    Args:
        file_name: The name of the file including the extension.
    
    Returns:
        loader: The corresponding document loader.
    """ 
    loaders = {
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader,
        '.csv': CSVLoader,
        '.txt': lambda fn: TextLoader(fn, encoding="utf-8"),
    }
    
    for extension, loader in loaders.items():
        if file_name.endswith(extension):
            return loader(file_name)
    
    st.error("Unsupported file type.")
    return None

def get_documents(loader, chunk_size, chunk_overlap):
    """
    Returns the split documents.
    
    Args:
        loader: Document loader.
        chunk_size: Size of the chunks.
        chunk_overlap: Overlap between chunks.

    Returns:
        splitted_documents: The list of split documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)
    return loader.load_and_split(text_splitter=text_splitter)

def get_vectorstore(doc_list):
    """
    Stores document embeddings in a vector store and returns it.

    Args:
        doc_list: The list of documents.

    Returns:
        vectorstore: The vector store containing document embeddings.
    """
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(doc_list, embeddings)

def get_retrievers(doc_list):
    """
    Creates and returns base retrievers.

    Args:
        doc_list: The list of documents.

    Returns:
        retrievers: A tuple containing sparse and dense retrievers.
    """
    k = 2
    bm25_retriever = BM25Retriever.from_documents(doc_list, kwargs={"k": k})
    faiss_retriever = st.session_state["vectorstore"].as_retriever(search_type="mmr", search_kwargs={"k": k, "score_threshold": 0.8})
    return bm25_retriever, faiss_retriever

def get_agent_executor():
    """
    Returns the agent executor object.

    Returns:
        agent_executor: The agent executor object.
    """
    search = TavilySearchResults(k=3)
    tool = create_retriever_tool(
        retriever=st.session_state["retrievers"]["compression_retriever"],
        name="search_documents",
        description="Searches and returns relevant excerpts from the uploaded documents."
    )
    
    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True, callbacks=[StreamHandler(st.empty())])
    agent_prompt = hub.pull("hwchase17/openai-functions-agent")
    
    agent = create_openai_functions_agent(llm, [search, tool], agent_prompt)
    return AgentExecutor(agent=agent, tools=[search, tool], verbose=True)

def initialize_session_state() -> None:
    """
    Initialize session state variables.
    
    """
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation = None

    if "message" not in st.session_state:
        st.session_state["message"] = []

    if "store" not in st.session_state:
        st.session_state["store"] = {}

    if "compressor" not in st.session_state:
        try:
            model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
            st.session_state["compressor"] = CrossEncoderReranker(model=model, top_n=3)
        except Exception as e:
            st.error(f"Error initializing compressor: {e}")
            logger.error(f"Error initializing compressor: {e}")

    if "retrievers" not in st.session_state:
        st.session_state["retrievers"] = None

# Streamlit page configuration
st.set_page_config(page_title="MyAssistant", page_icon="🤗")
st.title("🤗 MyAssistant")

# Greeting message
st.chat_message("assistant").write("*안녕하세요! 저는 당신의 비서입니다. 문서를 업로드한 후 질문을 입력해주세요* :sunglasses:")

# Initialize session state variables
initialize_session_state()

# Sidebar for user input
with st.sidebar:
    session_id = st.text_input("Session ID", value="Chating Room")
    uploaded_files = st.file_uploader("Upload files", type=['pdf', 'docx', 'csv', 'txt'], accept_multiple_files=True)

    if uploaded_files:
        doc_list = []
        for doc in uploaded_files:
            file_name = doc.name
            with open(file_name, "wb") as file:
                file.write(doc.getvalue())
                logger.info(f"Uploaded {file_name}")
            try:
                loader = get_loader(file_name)
                if loader:
                    splitted_documents = get_documents(loader, chunk_size=1000, chunk_overlap=50)
                    doc_list.extend(splitted_documents)
                    st.write("File has been uploaded!")
            except Exception as e:
                st.error(f"Error loading {file_name}: {e}")
                logger.error(f"Error loading {file_name}: {e}")

        # Initialize vector store and retrievers
        if "vectorstore" not in st.session_state and doc_list:
            try:
                st.session_state["vectorstore"] = get_vectorstore(doc_list)
            except Exception as e:
                st.error(f"Error initializing vector store: {e}")
                logger.error(f"Error initializing vector store: {e}")

        if "retrievers" in st.session_state:
            try:
                sparse_retriever, dense_retriever = get_retrievers(doc_list)
                st.session_state["retrievers"] = {"sparse_retriever": sparse_retriever, "dense_retriever": dense_retriever}

                ensemble_retriever = EnsembleRetriever(
                    retrievers=[sparse_retriever, dense_retriever], weights=[0.5, 0.5]
                )
                st.session_state["retrievers"]["ensemble_retriever"] = ensemble_retriever

                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=st.session_state["compressor"], 
                    base_retriever=ensemble_retriever
                )
                st.session_state["retrievers"]["compression_retriever"] = compression_retriever

            except Exception as e:
                st.error(f"Error initializing retriever: {e}")
                logger.error(f"Error initializing retriever: {e}")

    if st.button("Reset"):
        st.rerun()

# Chat history output
print_message()

# Chat logic
if user_input := st.chat_input("Please enter your message"):
    # Log user input
    st.chat_message("user").write(user_input)
    st.session_state["message"].append(ChatMessage(role="user", content=user_input))

    with st.chat_message("assistant"):
        if "retrievers" in st.session_state and "compressor" in st.session_state:
            try:
                # Initialize the retriever
                retriever = st.session_state["retrievers"]["ensemble_retriever"]
                relevant_docs = retriever.get_relevant_documents(user_input, kwargs={"k": 2})

                # Streaming output location
                stream_handler = StreamHandler(st.empty())

                # Define agent
                agent_executor = get_agent_executor()
                if "agent_executor" not in st.session_state:
                    st.session_state["agent_executor"] = agent_executor

                # Create agent with chat history
                agent_with_chat_history = RunnableWithMessageHistory(
                    agent_executor,
                    get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history"
                )
    
                # Generate response
                response = agent_with_chat_history.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": "MyTestSessionID"}}
                )
                answer = response["output"]

                # Display reference documents
                with st.expander("Reference Documents"):
                    for doc in relevant_docs:
                        st.markdown(doc.metadata['source'], help=doc.page_content)

                st.session_state["message"].append(ChatMessage(role="assistant", content=answer))
            except Exception as e:
                st.error(f"Error during processing: {e}")
                logger.error(f"Error during processing: {e}")
        else:
            st.error("Retriever is not initialized.")