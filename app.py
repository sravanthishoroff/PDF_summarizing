import io
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.llms.openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import AnalyzeDocumentChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain


pdf_docs = None
# function for extracting text 
def extract_text(pdf_content):
    text = ""
    pdf_file = io.BytesIO(pdf_content)
    try:
        pdf_reader = PdfReader(pdf_file)
        page_nums = len(pdf_reader.pages)
        for page_num in range(page_nums):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    except Exception as e:
        st.error(f"Error reading PDF {e}")
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

# contents of sidebar  
def sidebar():
    global pdf_docs
    st.sidebar.title('🤗💬 PDF chatbot')
    st.sidebar.subheader("Your Documents")
    pdf_docs = st.sidebar.file_uploader("upload your document here", type="pdf")
    if pdf_docs is not None:
        # displaying file details
        file_details = {"Name of the file":pdf_docs.name, "Size of the file":pdf_docs.size,"Type of the file":pdf_docs.type}
        st.sidebar.write("File Details:", file_details)

        try:
            pdf_content = pdf_docs.read()
            # Wrap the bytes in a BytesIO object to simulate a file-like object
            pdf_file = io.BytesIO(pdf_content)
            print(pdf_file)
        
        except Exception as e:
            st.sidebar.error(f"Error reading PDF {e}")
        return pdf_docs,pdf_content,pdf_file
    return pdf_docs
    

def main():    
    load_dotenv()
    st.title("🦜🔗Ask your pdf ")

    # extracting text from pdfs
    raw_text = extract_text(pdf_docs)      

    # summarizing the data from extracted text
    summarized_data = get_summarize_text(raw_text)

    # create chunks from the text
    chunked_data = get_text_chunks(raw_text)                

    # creating vectors from text
    vectorstore = get_vectorstore(chunked_data)            
    print(vectorstore)

    # Ask your question
    query = st.text_input("Ask your question here!!💬")
    print(query)
    if st.button("Process"):
        with st.spinner("Processing"):
            if query:
                docs = vectorstore.similarity_search(query=query, k=2)  
                chain = load_qa_chain(
                    llm = OpenAI(temperature=0),
                    chain_type="stuff"
                )
                response = chain.run(question=query,input_documents=docs)
                st.subheader("Processing Results...!")
                st.write(response)

if __name__ == "__main__":
    main()