import io
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.llms.openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import AnalyzeDocumentChain



def main():
    load_dotenv()
    st.set_page_config(page_title='🦜🔗 Text Summarization App')
    st.title("🦜🔗Ask your pdf ")
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("upload your document here", type="pdf")
        if st.button("Process"):
            with st.spinner("Processing"):
                if pdf_docs is not None:
                    # displaying file details
                    file_details = {"Name of the file":pdf_docs.name, "Size of the file":pdf_docs.size,"Type of the file":pdf_docs.type}
                    st.write("File Details:", file_details)

                    # displaying PDF file contents
                    pdf_content = pdf_docs.read()

                    # Wrap the bytes in a BytesIO object to simulate a file-like object
                    pdf_file = io.BytesIO(pdf_content)

                    # extracting text from pdfs
                    try:
                        text = ""
                        pdf_reader = PdfReader(pdf_file)
                        page_nums = len(pdf_reader.pages)
                        for page_num in range(page_nums):
                            page = pdf_reader.pages[page_num]
                            text += page.extract_text()
                        print(text)
                    except Exception as e:
                        st.error(f"Error reading PDF {e}")


                    # summarizing the data from extracted text
                    model = OpenAI(temperature=0)
                    summary_chain = load_summarize_chain(llm=model, chain_type='map_reduce')
                    summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain = summary_chain)
                    summarized_data = summarize_document_chain.run(text)
                    print(summarized_data)
                st.write(summarized_data)


if __name__ == "__main__":
    main()