# Logging library
from loguru import logger

# Streamlit
import streamlit as st

# Utility functions
from utills import (print_message, get_session_history, StreamHandler)

# LangChain Core
from langchain import hub
from langchain_core.messages import ChatMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

# LangChain OpenAI
from langchain_openai import ChatOpenAI

# Document loaders
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader, TextLoader, WebBaseLoader
from langchain_teddynote.document_loaders import HWPLoader
import bs4

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


def get_webbase_loader(urls):
    loader = WebBaseLoader(web_path=urls, encoding="utf-8")
    return loader

def get_document_loaders(file_name):
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
        '.txt': lambda fn: TextLoader(fn, encoding="utf-8"),
        '.hwp': HWPLoader
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
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

def initialize_session_state():
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
            st.session_state["compressor"] = None
            st.error(f"Error initializing compressor: {e}")
            logger.error(f"Error initializing compressor: {e}")

    if "retrievers" not in st.session_state:
        st.session_state["retrievers"] = None
        
    if "user_input" not in st.session_state:
        st.session_state["user_input"] = ""

def get_related_questions(answer):
    related_questions = generate_questions(answer)
    with st.expander("You might also be interested in:"):
        for question in related_questions:
            st.markdown(f"{question}")

def get_References(user_input):
    search = TavilySearchResults()
    search_response = search.invoke(user_input)

    relevant_docs = st.session_state["retrievers"]["dense_retriever"].get_relevant_documents(user_input)
    if relevant_docs:
        with st.expander("References"):
            st.subheader("Document", divider='gray')
            for index, doc in enumerate(relevant_docs):
                st.markdown(f"{doc.metadata['source']}", help=doc.page_content)
            st.subheader("Search", divider='gray')
            for index, doc in enumerate(search_response[:2]):
                st.markdown(f"{doc['url']}", help=doc['content'])

def generate_questions(answer):
    """
    Generate questions from the given text using OpenAI's GPT model.

    Args:
        text: The input text from which to generate questions.

    Returns:
        List of generated questions.
    """
    prompt = PromptTemplate.from_template(
        """
        You are an helpful assistant.
        Based on the given answer, generate follow-up questions. 
        You must 3 questions and length must under 50 characters.

        Answer: 
        {answer}

        Questions:
        """
    )
    ##
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)

    chain = prompt | llm

    response = chain.invoke(answer)
    following_questions = response.content.strip().split("\n")
    return following_questions

        
# Streamlit page configuration
st.set_page_config(page_title="MyAssistant", page_icon="🤗")
st.title("🤗 MyAssistant")

# Greeting message
st.chat_message("assistant").write("*안녕하세요! 저는 당신의 문서 작업 도우미입니다. 파일을 업로드 해주세요!* :sunglasses:")

# Initialize session state variables
initialize_session_state()

# Sidebar
with st.sidebar:
    session_id = st.text_input("Session ID", value="Chating Room")
    uploaded_files = st.file_uploader("Upload files", type=['pdf', 'docx', 'txt', 'hwp'], accept_multiple_files=True)
    url = st.text_input("Url")

    doc_list = []
    if uploaded_files or url:
        # Load files
        if uploaded_files:
            for doc in uploaded_files:
                file_name = doc.name
                with open(file_name, "wb") as file:
                    file.write(doc.getvalue())
                    logger.info(f"Uploaded {file_name}")
                try:
                    loader = get_document_loaders(file_name)
                    if loader:
                        splitted_documents = get_documents(loader, chunk_size=1000, chunk_overlap=50)
                        doc_list.extend(splitted_documents)
                except Exception as e:
                    st.error(f"Error loading {file_name}: {e}")
                    logger.error(f"Error loading {file_name}: {e}")
        # Load web pages
        if url:
            try:
                loader = get_webbase_loader(urls=url)
                if loader:
                    splitted_documents = get_documents(loader, chunk_size=1000, chunk_overlap=50)
                    doc_list.extend(splitted_documents)
            except:
                st.error(f"Error loading {url}: {e}")
                logger.error(f"Error loading {url}: {e}")      
        st.write("File has been uploaded!")

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


# initialize chat_input
chat_input = st.chat_input("Please enter your question:")
if chat_input:
    st.session_state["user_input"] = chat_input

# Chat logic
if st.session_state["user_input"]:
    user_input = st.session_state["user_input"]
    st.chat_message("user").write(user_input)
    st.session_state["message"].append(ChatMessage(role="user", content=user_input))

    with st.chat_message("assistant"):

        if st.session_state["retrievers"] and st.session_state["compressor"]:
            # Streaming output location
            stream_handler = StreamHandler(st.empty())
            try:
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
                    config={"configurable": {"session_id": "SessionID"}}
                )
                answer = response["output"]
    
                # Display reference documents
                get_References(user_input=user_input)

                # Display related questions
                get_related_questions(answer)

                st.session_state["message"].append(ChatMessage(role="assistant", content=answer))

            except Exception as e:
                st.error(f"Error during processing with Agent: {e}")
                logger.error(f"Error during processing with Agent: {e}")
        else:
            try:
                # chatprompt를 만들어야함.. 
                chat_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", "당신은 Question-Answering 챗봇입니다. 사용자의 질문에 대한 답변을 제공해주세요."),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", "#Question: \n{question}")
                    ]
                )
                llm = ChatOpenAI(model="gpt-4o-mini", streaming=True, callbacks=[StreamHandler(st.empty())])

                chain = chat_prompt | llm

                # Create agent with chat history
                chain_with_history = RunnableWithMessageHistory(
                    chain,
                    get_session_history,  # 세션 기록을 가져오는 함수
                    input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
                    history_messages_key="chat_history",  # 기록 메시지의 키
                )

                response = chain_with_history.invoke(
                    {"question": user_input},
                    config={"configurable": {"session_id": "SessionID"}}
                )
                answer = response.content

                # Display related questions
                get_related_questions(answer)
                
                st.session_state["message"].append(ChatMessage(role="assistant", content=answer))

            except Exception as e:
                st.error(f"Error during processing with LLM: {e}")
                logger.error(f"Error during processing with LLM: {e}")