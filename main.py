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

                # if not doc_list:
                #     raise ValueError("doc_list is empty. Make sure documents are loaded before initializing the retriever.")

                # (Sparse) bm25 retriever and (Dense) faiss retriever 를 초기화 합니다.
                bm25_retriever = BM25Retriever.from_documents(doc_list)
                bm25_retriever.k = k

                faiss_vectorstore = FAISS.from_documents(doc_list, OpenAIEmbeddings())
                faiss_retriever = faiss_vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "score_threshold": 0.8})

                # initialize the ensemble retriever
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

# Chat logic
if user_input := st.chat_input("메세지를 입력해 주세요"):
    # 사용자가 입력한 내용
    st.chat_message("user").write(f"{user_input}")
    st.session_state["message"].append(ChatMessage(role="user", content=user_input))

    # RAG
    if "retriever" in st.session_state:
        try:
            # Retriever 설정
            retriever = st.session_state["vectorstore"].as_retriever(search_type="mmr", search_kwargs={"k": 5, "score_threshold": 0.8})
            relevant_docs = retriever.get_relevant_documents(user_input)

            # filtered_docs = get_filtered_relevant_docs(user_input, relevant_docs)
            if not relevant_docs:
                st.warning("No relevant documents found.")
                documents_text = ""
            else:
                # 검색된 문서 내용을 문자열로 병합
                documents_text = "\n".join([doc.page_content for doc in relevant_docs])

            # LLM 응답 생성
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            rag_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system",
                     """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
                     If you don't know the answer, just say that you don't know. 
                     Use three sentences maximum and keep the answer concise.
                     If you get a question that is not related to the document, please ignore the context and answer it.
                     Please answer in korean.
                     
                     #Context:
                     {context} 
                     
                     #Answer:
                     """
                     ),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{question}"),
                    MessagesPlaceholder(variable_name='agent_scratchpad')
                ]
            )
            agent_prompt = hub.pull("hwchase17/openai-functions-agent")
            # agent 생성
            if "agent" not in st.session_state:
                search = TavilySearchResults(k=3)

                tool = create_retriever_tool(
                    retriever=retriever,  # 현재 세션의 retriever 사용
                    name="search_documents",  # 도구 이름
                    description="Searches and returns relevant excerpts from the uploaded documents."  # 도구 설명
                )
                tools = [search, tool]
                
                agent = create_openai_functions_agent(llm, tools, agent_prompt)
                agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
                st.session_state["agent_executor"] = agent_executor

            agent_executor = st.session_state["agent_executor"]
                # 채팅 메시지 기록을 관리하는 객체를 생성합니다.
            message_history = ChatMessageHistory()

            # 채팅 메시지 기록이 추가된 에이전트를 생성합니다.
            agent_with_chat_history = RunnableWithMessageHistory(
                agent_executor,
                # 대부분의 실제 시나리오에서 세션 ID가 필요하기 때문에 이것이 필요합니다
                # 여기서는 간단한 메모리 내 ChatMessageHistory를 사용하기 때문에 실제로 사용되지 않습니다
                # lambda session_id: message_history,
                get_session_history,
                # 프롬프트의 질문이 입력되는 key: "input"
                input_messages_key="input",
                # 프롬프트의 메시지가 입력되는 key: "chat_history"
                history_messages_key="chat_history"
                )
            response = agent_with_chat_history.invoke(
                {
                    "input": user_input
                },
                # 세션 ID를 설정합니다.
                # 여기서는 간단한 메모리 내 ChatMessageHistory를 사용하기 때문에 실제로 사용되지 않습니다
                config={"configurable": {"session_id": "MyTestSessionID"}},
            )
            answer = response["output"]
            

            # # 체인 생성
            # chain = prompt | llm

            # chain_with_memory = RunnableWithMessageHistory(
            #     chain,
            #     get_session_history,
            #     input_messages_key="question",
            #     history_messages_key="history",
            # )

            # # 응답 생성
            # response = chain_with_memory.invoke(
            #     {"context": documents_text, "question": user_input},
            #     config={"configurable": {"session_id": session_id}}
            # )
            # answer = response.content

            # AI의 답변 표시
            with st.chat_message("assistant"):
                st.write(answer)

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