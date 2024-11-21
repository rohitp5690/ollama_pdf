import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import RunnableWithMessageHistory
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_ollama import ChatOllama


os.environ['LANGCHAIN_API_KEY']=st.secrets('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2']='true'
os.environ['HF_TOKEN']=st.secrets('HF_TOKEN')
# GROQ_API_KEY=os.getenv('GROQ_API_KEY')


# llm_model=ChatGroq(model='Llama3-8b-8192',api_key=GROQ_API_KEY)
llm_model=ChatOllama(model='llama3.2:1b')
st.title('🧿CHAT WITH PDF🧿')
session_id=st.text_input('Session ID',value='default_session')
embeddings=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')


if 'my_container' not in st.session_state:
    st.session_state.my_container={}
    
uploaded_files=st.file_uploader('Upload PDF',accept_multiple_files=True,type='pdf')

if uploaded_files:
    documents=[]
    for uploaded_file in uploaded_files:
        temppdf=f'./temp.pdf'
        with open(temppdf,'wb') as file:
            file.write(uploaded_file.getvalue())
            file_name=uploaded_file.name
            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    splits=text_splitter.split_documents(documents=documents)
    vectorstore=FAISS.from_documents(documents=splits,embedding=embeddings)
    FAISS_Retriever=vectorstore.as_retriever()
    
    context_qna_system_prompt=(
            "given a chat history and the latest user question"
            'which might reference context in the chat history,'
            'formulate a standalone question which can be understood'
            'without the chat history. Do not answer the question,'
            'just reformulate it if needed and otherwise return it as is.'
    )
    context_qna_prompt=ChatPromptTemplate.from_messages([
        ('system',context_qna_system_prompt),
        MessagesPlaceholder(variable_name='chat_history'),
        ('user','{input}')
        ])

    History_Aware_Retriever=create_history_aware_retriever(llm=llm_model,retriever=FAISS_Retriever,prompt=context_qna_prompt)
    
    qna_system_prompt=(
        'you are an good assistant for question-answering tasks'
        'use the following pieces of retreived context to answer'
        'the question. if you dont know the answer then say you dont know'
        'use maximum 3 sentences, keep answer concise'
        '\n\n'
        '{context}'
    )

    qna_prompt=ChatPromptTemplate.from_messages([
        ('system',qna_system_prompt),
        MessagesPlaceholder(variable_name='chat_history'),
        ('user','{input}')
    ])

    QNA_Chain=create_stuff_documents_chain(llm=llm_model,prompt=qna_prompt)
    QNA_RAG_CHAIN=create_retrieval_chain(retriever=History_Aware_Retriever,combine_docs_chain=QNA_Chain)
    
    def session_history(session_id:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.my_container:
            st.session_state.my_container[session_id]=ChatMessageHistory()
        return st.session_state.my_container[session_id]

    Runnable_ragchain=RunnableWithMessageHistory(
        runnable=QNA_RAG_CHAIN,
        input_messages_key='input',
        output_messages_key='answer',
        history_messages_key='chat_history',
        get_session_history=session_history
    )

    user_input=st.text_input('Enter your query: ')
    if user_input:
        session_history_variable=session_history(session_id=session_id)
        response=Runnable_ragchain.invoke({'input':user_input},config={'configurable':{'session_id':session_id}})

        st.write(st.session_state.my_container)
        st.write('Assistant: \n',response['answer'])
        st.write('Chat History: \n',session_history_variable.messages)
        




