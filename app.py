import io
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.llms.openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import AnalyzeDocumentChain
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from htmlTemplates import css, bot_template, user_template
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain



# function for extracting text 
def extract_text(pdf_file):
    text = ""
    try:
        pdf_reader = PdfReader(pdf_file)
        page_nums = len(pdf_reader.pages)
        if page_nums > 0:
            for page_num in range(page_nums):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        else:
            st.warning("The PDF file has no pages.")
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

# function for summarizing the text
def get_summarize_text(text):
    model = OpenAI(temperature=0)
    summary_chain = load_summarize_chain(llm=model, chain_type='map_reduce')
    summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain = summary_chain)
    summarized_data = summarize_document_chain.run(text)
    return summarized_data

# function for chunking the data
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# function for creating embeddings
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks,embedding=embeddings)
    return vectorstore 

# function for conversation memory buffer
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key = 'chat history',
        return_messages= True)
    conversation_chain = load_qa_chain(llm = llm,
                                       retriever=vectorstore,
                                       memory=memory)   

    return conversation_chain   

# handle user_input
def handle_userinput(user_question):
    if st.session_state.conversation is not None:
        response = st.session_state.conversation({'question':user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        st.error("Conversation chain is not initialized. Please upload a PDF file first.")

# main function
def main():    
    load_dotenv()
    
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)


    with st.sidebar:    
        st.subheader("Your documents")

        pdf_docs = st.file_uploader("upload your document here", type="pdf")

        if pdf_docs is not None:
            # extracting text from pdfs
            raw_text = extract_text(pdf_docs)
            if not raw_text:
                    st.warning("No text extracted from the PDF. Please upload a valid PDF file.")
                    st.stop()    


            # displaying file details
            file_details = {"Name of the file":pdf_docs.name, "Size of the file":pdf_docs.size,"Type of the file":pdf_docs.type}
            st.write("File Details:", file_details)      
        
                
            # summarizing the data from extracted text
            summarized_data = get_summarize_text(raw_text)

            # create chunks from the text
            chunked_data = get_text_chunks(raw_text)             

            # creating vectors from text
            vectorstore = get_vectorstore(chunked_data)            
            print(vectorstore)

            # if vectorstore:
            #     st.success("Text extraction and vectorization completed successfully!")

            # Ask your question
            # query = st.text_input("Ask your question here!!💬")     

            # if st.button("Process"):
            #     with st.spinner("Processing"):
            #         if query:
            #             docs = vectorstore.similarity_search(query=query, k=2)  
            #             chain = load_qa_chain(
            #                 llm = OpenAI(temperature=0),
            #                 chain_type="stuff"
            #             )
            # with get_openai_callback() as cost:
            #     response = chain.run(input_documents=docs, question=query)
            #     print(cost)



            # create conversation chain
            st.session_state.conversation = get_conversation_chain(
                vectorstore)

            # response = chain.run(question=query,input_documents=docs)
            # st.subheader("Processing Results...!")
            # st.write(response)

            cancel_button = st.button('Cancel')
            if cancel_button:
                st.stop()

if __name__ == "__main__":
    main()