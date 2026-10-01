import pypdf
import langchain_community.document_loaders
FILE_PATH = "Data/rag_practice_sample.pdf"
def load_pdf(file_path):
    loader = langchain_community.document_loaders.PyPDFLoader(file_path)
    documents = loader.load()  
    print("Document objects:", len(documents))
    print("Document objects:", documents[0].page_content)
    print("Document objects:", documents[0].metadata)
    return documents
if __name__ == "__main__":
    load_pdf(FILE_PATH)