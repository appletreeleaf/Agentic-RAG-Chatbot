# Logging library
from loguru import logger

# Streamlit
import streamlit as st

# Utility
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
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
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
    문서 형식에 맞는 loader를 반환합니다.

    args:
    file_name: 확장자를 포함한 파일 이름
    
    return : loader
    """ 

    if file_name.endswith('.pdf'):
        loader = PyPDFLoader(file_name)
    elif file_name.endswith('.docx'):
        loader = Docx2txtLoader(file_name)
    elif file_name.endswith('.csv'):
        loader = CSVLoader(file_name)
    elif file_name.endswith('.txt'):
        loader = TextLoader(file_name, encoding="utf-8")
    else:
        return st.error("Unsupported file type.")
    
    return loader

def get_documents(loader, chunk_size, chunk_overlap):
    """
    분할된 문서를 반환합니다.
    
    return : 분할된 문서
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splitted_documents = loader.load_and_split(text_splitter=text_splitter)
    return splitted_documents

def get_vectorstore(doc_list):
    """
    문서의 임베딩 값을 벡터 저장소에 저장하여 반환합니다.

    args:
    doc_list : 문서 집합

    return : 벡터 저장소
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(doc_list, embeddings)
    return vectorstore

def get_retrievers(doc_list):
    """
    base retriever를 생성하여 반환합니다.

    args:
    doc_list : 문서 집합

    base retriever : (dense_retriever, sparse retriever)

    return : base retriever
    """
    k = 2
    # sparse retriever
    bm25_retriever = BM25Retriever.from_documents(doc_list, kwargs={"k": k})
    # dense retriever
    faiss_retriever = st.session_state["vectorstore"].as_retriever(search_type="mmr", search_kwargs={"k": k, "score_threshold": 0.8})
    retrievers = (bm25_retriever, faiss_retriever)
    return retrievers

def get_agent_excutor():
    """
    agent_executor 객체를 반환합니다.

    return : agent_executor 객체
    """
    search = TavilySearchResults(k=3)

    tool = create_retriever_tool(
        retriever=st.session_state["compression_retriever"],
        name="search_documents",
        description="Searches and returns relevant excerpts from the uploaded documents."
    )
    tools = [search, tool]
    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True, callbacks=[stream_handler])
    agent_prompt = hub.pull("hwchase17/openai-functions-agent")
    
    agent = create_openai_functions_agent(llm, tools, agent_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor


st.set_page_config(page_title="MyAssistant", page_icon="🤗")
st.title("🤗 MyAssistant")

st.chat_message("assistant").write("*안녕하세요! 저는 당신의 비서입니다. 문서를 업로드한 후 질문을 입력해주세요* :sunglasses:")
# chain을 저장할 상태 변수        
if "conversation_chain" not in st.session_state:
    st.session_state.conversation = None

# 메세지 내용을 기록하는 상태 변수
if "message" not in st.session_state:
    st.session_state["message"] = [] 

# 채팅 기록을 저장하는 상태 변수
if "store" not in st.session_state:
    st.session_state["store"] = dict()

if "compressor" not in st.session_state:
    st.session_state["compressor"] = None
    try:
        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
        compressor = CrossEncoderReranker(model=model, top_n=3)
        st.session_state["compressor"] = compressor
    except Exception as e:
        st.error(f"Error initializing compressor: {e}")
        logger.error(f"Error initializing compressor: {e}")

# retriever를 저장하는 상태 변수
if "retrievers" not in st.session_state:
    st.session_state["retrievers"] = None

# 사이드
with st.sidebar:
    session_id = st.text_input("Session ID", value="Chating Room")
    uploaded_file = st.file_uploader("Upload files", type=['pdf', 'docx', 'csv', 'txt'], accept_multiple_files=True)

    if uploaded_file:
        doc_list = []
        for doc in uploaded_file:
            file_name = doc.name
            with open(file_name, "wb") as file:
                file.write(doc.getvalue())
                logger.info(f"Uploaded {file_name}")
            try:
                loader = get_loader(file_name)
                splitted_documents = get_documents(loader, chunk_size=1000, chunk_overlap=50)
                doc_list.extend(splitted_documents)
                st.write("File has been uploaded!")
            except Exception as e:
                st.error(f"Error loading {file_name}: {e}")
                logger.error(f"Error loading {file_name}: {e}")

        # RAG 관련 변수 초기화
        if "vectorstore" not in st.session_state:
            if doc_list:
                try:
                    vectorstore = get_vectorstore(doc_list)
                    st.session_state["vectorstore"] = vectorstore
                except Exception as e:
                    st.error(f"Error initializing vector store: {e}")
                    logger.error(f"Error initializing vector store: {e}")

        if "retrievers" in st.session_state:
            try:
                sparse_retriever, dense_retriever = get_retrievers(doc_list)
                st.session_state["retrievers"] = {"sparse_retriever": sparse_retriever, "dense_retriever": dense_retriever}

                ensemble_retriever = EnsembleRetriever(
                retrievers=[sparse_retriever, dense_retriever], weights=[0.5, 0.5])
                st.session_state["retrievers"]["ensemble_retriever"] = ensemble_retriever

                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=st.session_state["compressor"], 
                    base_retriever=st.session_state["retrievers"]["ensemble_retriever"]
                )
                st.session_state["retrievers"]["compression_retriever"] = compression_retriever

            except Exception as e:
                st.error(f"Error initializing retriever: {e}")
                logger.error(f"Error initializing retriever: {e}")

    if st.button("Reset"):
        st.rerun()


# 채팅 기록을 출력
print_message()

# chat logic
if user_input := st.chat_input("메세지를 입력해 주세요"):
    # 유저 메세지 입력
    st.chat_message("user").write(f"{user_input}")
    st.session_state["message"].append(ChatMessage(role="user", content=user_input))

    with st.chat_message("assistant"):
        # RAG
        if "retrievers" in st.session_state and "compressor" in st.session_state:
            try:
                # 관련 문서 초기화
                retriever = st.session_state["retrievers"]["ensemble_retriever"]
                relevant_docs = retriever.get_relevant_documents(user_input)

                # streaming 출력 위치
                stream_handler = StreamHandler(st.empty())

                # agent 정의
                agent_executor = get_agent_excutor()
                # session에 저장
                if "agent_excutor" not in st.session_state:
                    st.session_state["agent_executor"] = agent_executor

                # 채팅 메시지 기록이 추가된 에이전트를 생성합니다.
                agent_with_chat_history = RunnableWithMessageHistory(
                    agent_executor,
                    get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history"
                    )
    
                # 응답 생성
                response = agent_with_chat_history.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": "MyTestSessionID"}}
                )
                answer = response["output"]
                # 참고 문서 표시
                with st.expander("참고 문서"):
                    for doc in relevant_docs:
                        st.markdown(doc.metadata['source'], help=doc.page_content)

                st.session_state["message"].append(ChatMessage(role="assistant", content=answer))
            except Exception as e:
                st.error(f"Error during processing: {e}")
                logger.error(f"Error during processing: {e}")
        else:
            st.error("Retriever is not initialized.")