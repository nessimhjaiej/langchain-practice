from langchain_text_splitters import RecursiveCharacterTextSplitter
import loader
FILE_PATH = "Data/rag_practice_sample.pdf"
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=50,
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split documentation into {len(split_docs)} chunks.")
    print(f"first 2 chunks data : \n {split_docs[0].page_content} \n \n AND  {split_docs[1].page_content}")   
    return split_docs

if __name__ == "__main__" :
    documents = loader.load_pdf(FILE_PATH) 
    chunks = split_text(documents)

