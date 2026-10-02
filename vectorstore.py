from dotenv import load_dotenv 
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import splitter
import loader
def create_vectorstore(documents):
    chunks = splitter.split_text(documents)
    load_dotenv() 
    embedding_model = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local("vectorstore") 
    return vectorstore

if __name__ == "__main__":
    documents = loader.load_pdf("Data/rag_practice_sample.pdf")
    vectorstore = create_vectorstore(documents)
    print("Vectorstore created and saved locally.")