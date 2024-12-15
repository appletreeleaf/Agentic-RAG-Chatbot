from loguru import logger
import streamlit as st
from utills import print_message, get_session_history, StreamHandler

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

st.set_page_config(page_title="MyAssistant", page_icon="🤗")
st.title("🤗 MyAssistant")

if "conversation_chain" not in st.session_state:
    st.session_state.conversation = None

# 사이드 바
with st.sidebar:
    session_id = st.text_input("Session ID", value="Chating Room")
    uploaded_file = st.file_uploader("Upload a file", type=['pdf', 'docx', 'csv', 'txt'], accept_multiple_files=True)

    if uploaded_file is not None:
        doc_list = []
        for doc in uploaded_file:
            file_name = doc.name
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
                
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
                splitted_documents = text_splitter.split_documents(documents)
                doc_list.extend(splitted_documents)
            
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

                if not doc_list:
                    raise ValueError("doc_list is empty. Make sure documents are loaded before initializing the retriever.")

                # (Sparse) bm25 retriever and (Dense) faiss retriever 를 초기화 합니다.
                bm25_retriever = BM25Retriever.from_documents(doc_list)
                bm25_retriever.k = k

                faiss_vectorstore = FAISS.from_documents(doc_list, OpenAIEmbeddings())
                faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": k})

                # initialize the ensemble retriever
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
                )
                st.session_state["retriever"] = ensemble_retriever
            except Exception as e:
                st.error(f"Error initializing retriever: {e}")
                logger.error(f"Error initializing retriever: {e}")

    if st.button("Reset"):
        st.session_state["message"] = []
        st.rerun()

# 메세지 내용을 기록하는 상태 변수
if "message" not in st.session_state:
    st.session_state["message"] = [] 

# 채팅 기록을 저장하는 상태 변수
if "store" not in st.session_state:
    st.session_state["store"] = dict()

# 채팅 기록을 출력하는 부분을 함수화
print_message()

# initialize chat box
if user_input := st.chat_input("메세지를 입력해 주세요"):
    # 사용자가 입력한 내용
    st.chat_message("user").write(f"{user_input}")
    st.session_state["message"].append(ChatMessage(role="user", content=user_input))

    # RAG 기능: 사용자의 질문에 대한 관련 문서 검색
    if "retriever" in st.session_state:
        try:
            # 유사도 높은 K 개의 문서를 검색합니다.
            k = 2

            relevant_docs = st.session_state["vectorstore"].similarity_search(user_input, k=k)
            if not relevant_docs:
                st.warning("관련 문서를 찾을 수 없습니다.")
                st.stop()

            # 검색된 문서 내용을 문자열로 변환
            documents_text = "\n".join([doc.page_content for doc in relevant_docs])

            # LLM 응답 생성
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            prompt = hub.pull("rlm/rag-prompt")

            # context를 Runnable 형식으로 래핑
            context = {"context": documents_text, "question": user_input}

            # 체인 생성
            rag_chain = (
                prompt | llm
            )

            chain_with_memory = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="question",
                history_messages_key="history",
            )

            response = chain_with_memory.invoke(
                context,
                config={"configurable": {"session_id": session_id}}
            )
            answer = response.content

            # AI의 답변
            with st.chat_message("assistant"):
                stream_handler = StreamHandler(st.empty())
                llm = ChatOpenAI(model="gpt-4o-mini", streaming=True, callbacks=[stream_handler])
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", "사용자의 질문에 짧고 간결한 답변을 생성해주세요."),
                        MessagesPlaceholder(variable_name="history"),
                        ("human", "{question}")  # 검색된 문서 내용 추가
                    ]
                )
                chain = prompt | llm
                chain_with_callbacks = RunnableWithMessageHistory(
                    chain,
                    get_session_history,
                    input_messages_key="question",
                    history_messages_key="history",
                )
                response = chain_with_callbacks.invoke(
                    {"question": user_input},
                    config={"configurable": {"session_id": session_id}}
                )

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