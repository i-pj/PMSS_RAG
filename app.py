import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

PDF = "./FAQs PMSSS 2023-24.pdf"


@st.cache_resource
def load_process_and_create_vector_store():
    loader = PyPDFLoader(PDF)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(
        model_kwargs={"tokenizer_kwargs": {"clean_up_tokenization_spaces": True}}
    )
    vector_store = FAISS.from_texts([text.page_content for text in texts], embeddings)
    return vector_store


def setup_qa_chain(vector_store):
    llm = Ollama(model="llama3:8b")
    template = """
    You are an AI assistant for the PMSSS scholarship program. Use the following pieces of context to answer the human's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    
    Human: {question}
    AI Assistant: Let me help you with that.
    """
    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    return qa_chain


def main():
    st.title("PMSSS Scholarship Assistant")

    vector_store = load_process_and_create_vector_store()
    qa_chain = setup_qa_chain(vector_store)

    question = st.text_input("How can I assist you with the PMSSS scholarship program?")

    if question:
        with st.spinner("Searching for information..."):
            result = qa_chain.invoke({"query": question})

        st.write(result["result"])

        with st.expander("View sources"):
            for i, source in enumerate(result["source_documents"], 1):
                st.write("Source " + str(i) + ":")
                st.write(source.page_content[:500] + "...")


if __name__ == "__main__":
    main()
