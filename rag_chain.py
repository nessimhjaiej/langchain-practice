import dotenv
import langchain_openai
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS


def chain():
    dotenv.load_dotenv()
    embedding_model = langchain_openai.OpenAIEmbeddings()
    vectorstore = FAISS.load_local("vectorstore", embedding_model, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
    llm = langchain_openai.ChatOpenAI(model="gpt-5-mini")
    system_message = (
        "You are a helpful assistant that answers using only the provided context. "
        "If the answer is not in the context, say \"I don't know\".\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{input}")
    ])
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    response = retrieval_chain.invoke({"input": "What is the main topic of the document?"})
    print("Chain invoked successfully.")
    print(response["answer"])
    print(f"chunks retrieved {len(response['context'])} \n")


if __name__ == "__main__":
    chain()
