import random
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub, ctransformers
from sentence_transformers import SentenceTransformer
from langchain.chains import VectorDBQAWithSourcesChain
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import os
import tensorflow as tf
import time
from PIL import Image
# import fitz
import bitsandbytes as bnb
# from langchain_community.llms import GPT4All
# from langchain_community.llms  import GPT4AllGPU

from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import GPT4All
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader



class CustomLLM:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_response(self, input_text, max_length=50):
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        output = self.model.generate(input_ids, max_length=max_length)
        response_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response_text

    def save_model(self, path):
        self.model.save_pretrained(path)

    def save_tokenizer(self, path):
        self.tokenizer.save_pretrained(path)



# Reading PDF file
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# def get_pdf_images(pdf_docs):
#     # Creating directory for Images
#     if not os.path.exists('images'):
#         os.makedirs('images')

#     for pdf in pdf_docs:
#         #Define path to PDF file
#         # file_path = 'dataset/SolfaInterfaceDocument.pdf'

#         #Open PDF file using Fitz module
#         pdf_file = fitz.open(stream=pdf.read())
        
#         #Calculate the number of pages in PDF file
#         page_nums = len(pdf_file)

#         #Create an empty list to store image information
#         images_list = []


    #     #Extract all images information from each page
    #     for page_num in range(page_nums):
    #         page_content = pdf_file[page_num]
    #         images_list.extend(page_content.get_images())

    #     #Raise error if PDF has no images
    #     if len(images_list)==0:
    #         raise ValueError('No images found')

    #     #Extract and save the Images
    #     for i, image in enumerate(images_list,start=1):
    #         #Extract the image object number
    #         xref = image[0]
    #         #Extract image
    #         base_image = pdf_file.extract_image(xref)
    #         #Store image bytes
    #         image_bytes = base_image['image']
    #         #Store image extention
    #         image_ext = base_image['ext']
    #         #Generate image file name
    #         image_name = str(i)+ '.'+image_ext
    #         #Save image
    #         with open(os.path.join('images',image_name),'wb') as image_file:
    #             image_file.write(image_bytes)
    #             image_file.close()

    # return


# splitting texts into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=0,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Creating a vector store (or vector database) of embeddings for a given set of text chunks using FAISS by Facebook.
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# using HuggingFaceHub get 'llm' model remotly
def get_Hub_llm():

    llm = HuggingFaceHub( 
        # repo_id="google/flan-ul2",
        repo_id="HuggingFaceH4/zephyr-7b-beta", 
                         model_kwargs={"temperature":0.1, 
                                       "max_length":2048, 
                                        "top_k":50,
                                        "task":"text-generation",
                                        "num_return_sequences":3,
                                       "top_p":0.95})
    # llm.save_pretrained("llmLocalHub/model")
    # print(type(llm)) ==> <class 'langchain.llms.huggingface_hub.HuggingFaceHub'>
    # AutoModel.save(llm)
    # AutoTokenizer.save()
    return llm



#load the llm model locally
def get_local_llm():
    gpt4all = GPT4All(
    model="D:\\Projects\\Sage\\gpt4AllTest\\mistral-7b-openorca.Q4_0.gguf",
    max_tokens=10000,
    temp=0.1,
    top_k=40, top_p=0.4, repeat_penalty=1.18, repeat_last_n=64, n_batch=80, n_predict=10000, streaming=True
    )
    # gpt4all = GPT4AllGPU(
    # model="D:\\Projects\\Sage\\gpt4AllTest\\mistral-7b-openorca.Q4_0.gguf",
    # max_tokens=10000,
    # )
    # return model
    return gpt4all




# create conversation chain
def get_conversation_chain(vectorstore,llm):
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type = "stuff",
        verbose=True,
        # retriever=vectorstore.as_retriever(search_kwargs = {"k" : 3, "search_type" : "similarity"}),
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    count = 1
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            with st.chat_message("User", avatar='👨‍💻'):
                st.write(message.content)
        else:
            with st.chat_message("assistant", avatar='static/sage-icon-2.png'):
                # Check if the message is the latest one
                if i == len(st.session_state.chat_history) - 1:
                    # Placeholder for the new message
                    t = st.empty()
                    

                    print("======Result======")
                    print(message.content)
                    print("==================")
                    # Typing effect for the new message
                    for char_index in range(len(message.content) + 1):
                        t.write("%s..." % message.content[0:char_index])
                        time.sleep(0.009)
                    # st.write(message.content)
                else:
                    # For older messages, display normally
                    st.write(message.content)
                # Add thumbs-up and thumbs-down buttons without affecting the chat history
                col1, col2, col3, col4 = st.columns([3, 3, 0.5, 0.5])
                with col3:
                    if st.button("👍", key=f"thumbs_up{count}"):
                        st.balloons()  # Show balloons effect
                    count+=1
                with col4:
                    if st.button(":thumbsdown:", key=f"thumbs_down{count}"):
                        st .write("Dislike")  # Show dislike message
    return               

# def getChromaVectorstore(path):
#     # loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
#     # data = loader.load()

#     loader =PyPDFLoader(path)
#     data=loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
#     all_splits = text_splitter.split_documents(data)
#     vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())
#     return vectorstore

def getChromaVectorstore(pathList):
    file_splits =[]
    for path in pathList:
        loader =PyPDFLoader(path)
        data=loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        file_splits.extend(text_splitter.split_documents(data))
    vectorstore = Chroma.from_documents(documents=file_splits, embedding=GPT4AllEmbeddings())
    return vectorstore

# Sage rondom greetings
def get_random_greeting():
    greetings = [
        "Hello!👋, I'm Vodafone Sage, your AI assistant. How can I assist you today?",
        "Greetings! I'm Vodafone Sage, here to help you with APIs, descriptions, test cases, and more 🥰. What do you need assistance with?",
        "Hi👋, there! Vodafone Sage at your service. What can I do for you today?",
        "Welcome!👋, I'm Vodafone Sage, ready to support you with any work-related queries. How can I help you today?",
        "Good day! I'm Vodafone Sage, your go-to assistant 🤖 for all things work-related. Let me know what you're looking for!"
    ]

    return random.choice(greetings)
    
def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

def run():
    load_dotenv()

    st.set_page_config(page_title="Vodafone Sage",
                       page_icon="static/sage-icon-2.png")
    
    
    # Custom HTML/CSS for the banner
    custom_html = """
    <style>
        .st-emotion-cache-1avcm0n{
            background-color: rgb(162 23 18);
        }
        button[title="View fullscreen"]{
            visibility: hidden;
        }
    </style>
    """

    # Display the custom HTML
    st.markdown(custom_html,unsafe_allow_html=True)

    



    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    st.image('static/sage-icon-5.png', width=400)
    st.header("Vodafone Sage")
    # st.snow()
    # st.toast('Your edited image was saved!', icon='😍')
    # st.balloons()
    
    
    with st.chat_message("assistant", avatar='static/sage-icon-2.png'):
        st.write(get_random_greeting())

    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    # Sidebar content
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                if not os.path.exists("documents"):
                    os.makedirs("documents")
                pathList = []
                for uploaded_file in pdf_docs:
                    with open(os.path.join("documents",uploaded_file.name),"wb") as f:
                        f.write(uploaded_file.getbuffer())
                        path = os.path.join("documents", uploaded_file.name)
                        pathList.append(path)
                        print(path)
                print(pathList)        


                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                
                
                
                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # extract images
                # get_pdf_images(pdf_docs)

                print("befor vector store")
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # create vector store
                # vectorstore = get_vectorstore(text_chunks)
                # vectorstore = getChromaVectorstore("D:\\Projects\\Sage\\queryDoc\\Solfa_Interface_Document.pdf")
                vectorstore = getChromaVectorstore(pathList)

                # print("vector store: ")
                # print(vectorstore)
                # # create llm
                llm = get_local_llm()
                # llm=get_Hub_llm()



                # Use HuggingFaceHub to create the conversation chain
                # llm = get_Hub_llm()
                st.session_state.conversation = get_conversation_chain(vectorstore, llm)


run()
# get_local_llm()