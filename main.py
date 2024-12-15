from loguru import logger
import streamlit as st
from utills import (print_message, get_session_history, StreamHandler,
                    get_filtered_relevant_docs)

from langchain import hub
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_teddynote.messages import stream_response

st.set_page_config(page_title="MyAssistant", page_icon="🤗")
st.title("🤗 MyAssistant")

if "conversation_chain" not in st.session_state:
    st.session_state.conversation = None

# side bar
with st.sidebar:
    session_id = st.text_input("Session ID", value="Chating Room")
    uploaded_file = st.file_uploader("Upload a file", type=['pdf', 'docx', 'csv', 'txt'], accept_multiple_files=True)

    if uploaded_file:
        # 업로드된 파일을 저장
        doc_list = []
        for doc in uploaded_file:
            file_name = doc.name
            # 파일을 write binary(wb)모드로 열기
            with open(file_name, "wb") as file:
                file.write(doc.getvalue())
                logger.info(f"Uploaded {file_name}")
            try:
                if file_name.endswith('.pdf'):
                    loader = PyPDFLoader(file_name)
                elif file_name.endswith('.docx'):
                    loader = Docx2txtLoader(file_name)
                elif file_name.endswith('.csv'):
                    loader = CSVLoader(file_name)
                elif file_name.endswith('.txt'):
                    loader = TextLoader(file_name, encoding="utf-8")
                else:
                    st.error("Unsupported file type.")
                    continue
                
                # 문서를 load and split -> chunk
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
                splitted_documents = text_splitter.split_documents(documents)
                doc_list.extend(splitted_documents)

                st.write("File has been uploaded!")
            except Exception as e:
                st.error(f"Error loading {file_name}: {e}")
                logger.error(f"Error loading {file_name}: {e}")

        # RAG 관련 변수 초기화
        if "vectorstore" not in st.session_state:
            if doc_list:
                try:
                    embeddings = OpenAIEmbeddings()
                    st.session_state["vectorstore"] = FAISS.from_documents(doc_list, embeddings)
                except Exception as e:
                    st.error(f"Error initializing vector store: {e}")
                    logger.error(f"Error initializing vector store: {e}")

        if "retriever" not in st.session_state:
            try:
                # 유사도 높은 K 개의 문서를 검색합니다.
                k = 2
                # bm25 retriever(Sparse) and faiss retriever(Dense)를 초기화 합니다.
                bm25_retriever = BM25Retriever.from_documents(doc_list)
                bm25_retriever.k = k

                faiss_vectorstore = FAISS.from_documents(doc_list, OpenAIEmbeddings())
                faiss_retriever = faiss_vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "score_threshold": 0.8})

                ensemble_retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
                )
                st.session_state["retriever"] = ensemble_retriever
            except Exception as e:
                st.error(f"Error initializing retriever: {e}")
                logger.error(f"Error initializing retriever: {e}")

    if st.button("Reset"):
        # st.session_state["message"] = []
        st.rerun()

# 메세지 내용을 기록하는 상태 변수
if "message" not in st.session_state:
    st.session_state["message"] = [] 

# 채팅 기록을 저장하는 상태 변수
if "store" not in st.session_state:
    st.session_state["store"] = dict()

# 채팅 기록을 출력하는 부분을 함수화
print_message()

# chat logic
if user_input := st.chat_input("메세지를 입력해 주세요"):
    # 사용자가 입력한 내용
    st.chat_message("user").write(f"{user_input}")
    st.session_state["message"].append(ChatMessage(role="user", content=user_input))

    with st.chat_message("assistant"):
        # RAG
        if "retriever" in st.session_state:
            try:
                # Retriever 설정
                retriever = st.session_state["vectorstore"].as_retriever(search_type="mmr", search_kwargs={"k": 2, "score_threshold": 0.8})
                relevant_docs = retriever.get_relevant_documents(user_input)

                # LLM 응답 생성
                # agent 정의
                search = TavilySearchResults(k=3)

                tool = create_retriever_tool(
                    retriever=retriever,  # 현재 세션의 retriever 사용
                    name="search_documents",  # 도구 이름
                    description="Searches and returns relevant excerpts from the uploaded documents."  # 도구 설명
                )
                tools = [search, tool]

                # streaming 출력 자리
                stream_handler = StreamHandler(st.empty())
                llm = ChatOpenAI(model="gpt-4o-mini", streaming=True, callbacks=[stream_handler])
                
                agent_prompt = hub.pull("hwchase17/openai-functions-agent")
                agent = create_openai_functions_agent(llm, tools, agent_prompt)

                agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
                # session에 저장
                if "agent_excutor" not in st.session_state:
                    st.session_state["agent_executor"] = agent_executor

                agent_executor = st.session_state["agent_executor"]
                # 채팅 메시지 기록이 추가된 에이전트를 생성합니다.
                agent_with_chat_history = RunnableWithMessageHistory(
                    agent_executor,
                    get_session_history,    # chat history를 불러옴
                    input_messages_key="input",
                    # 프롬프트의 메시지가 입력되는 key: "chat_history"
                    history_messages_key="chat_history"
                    )
    
                # AI의 답변 표시
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