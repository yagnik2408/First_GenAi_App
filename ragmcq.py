import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
import tempfile
from dotenv import load_dotenv
load_dotenv()

# Title & Instructions
#st.set_page_config(page_title="RAG MCQ & QA Generator", layout="centered")
st.title("RAG-based MCQ & Short Answer Generator")
st.markdown("Upload a GenAI-related PDF and generate exam-style questions using Retrieval-Augmented Generation (RAG).")

# PDF Upload
uploaded_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])

# Question Type Selection
question_type = st.radio("Select Question Type to Generate:", ["Both", "MCQ Only", "Short Answers Only"])

if uploaded_file:
    with st.spinner("Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        if not documents:
            st.error("❌ No content extracted from the PDF. Please try another file.")
            st.stop()

        # Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = splitter.split_documents(documents)

        if not texts:
            st.error("❌ Text splitting failed. Try a different PDF.")
            st.stop()

        # Load embeddings
        embeddings = AzureOpenAIEmbeddings(
            openai_api_base=os.getenv("EMBEDDING_AZURE_OPENAI_API_BASE"),
            azure_endpoint=os.getenv("EMBEDDING_AZURE_OPENAI_API_ENDPOINT"),
            openai_api_version=os.getenv("EMBEDDING_AZURE_OPENAI_API_VERSION"),
            openai_api_key=os.getenv("EMBEDDING_AZURE_OPENAI_API_KEY"),
            deployment=os.getenv("EMBEDDING_AZURE_OPENAI_DEPLOYMENT_NAME"),
            model="text-embedding-3-large",
            chunk_size=10
        )

        # FAISS vector store
        db = FAISS.from_documents(texts, embeddings)
        retriever = db.as_retriever()

        # Azure LLM
        llm = AzureChatOpenAI(
            openai_api_base=os.getenv("AZURE_OPENAI_API_BASE"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            model_name="gpt-4o",
            temperature=0.5,
            model_kwargs={"top_p": 0.9, "max_tokens": 1500}
        )


        # QA Chain
        qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff"
        )

        # Prompts
        mcq_prompt = """
        Based on the provided content, generate 5 multiple choice questions.
        For each question, provide:
        - The question
        - 4 options (A, B, C, D)
        - The correct answer with explanation.
        """

        short_ans_prompt = """
        Based on the provided content, generate 5 short answer questions and their correct answers.
        For each, provide:
        - The question
        - The answer
        """

        # Generate Questions
        with st.spinner("Generating questions..."):
            if question_type in ["Both", "MCQ Only"]:
                mcq_result = qa_chain.invoke(mcq_prompt)
                with st.expander("Generated MCQs"):
                    st.markdown(f"```markdown\n{mcq_result['answer']}\n```")

            if question_type in ["Both", "Short Answers Only"]:
                short_ans_result = qa_chain.invoke(short_ans_prompt)
                with st.expander("Generated Short Answer Questions"):
                    st.markdown(f"```markdown\n{short_ans_result['answer']}\n```")

        st.success("✅ Question Generation Completed!")
