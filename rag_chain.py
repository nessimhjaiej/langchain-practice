import dotenv
import langchain_openai
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

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
    contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question, "
                "rephrase it into a standalone question that can be "
                "understood without the chat history. Do NOT answer it, "
                "just reformulate it if needed, otherwise return it as is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)
    combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, combine_docs_chain)
    chat_history = []
    print("Ask questions about the document. Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        response = rag_chain.invoke({"input": question, "chat_history": chat_history})
        print(f"Bot: {response['answer']}")
        print(f"(chunks retrieved: {len(response['context'])})\n")
        chat_history.extend([HumanMessage(question), AIMessage(response["answer"])])


if __name__ == "__main__":
    chain()
