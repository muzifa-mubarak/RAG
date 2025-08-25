from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


embedd=GoogleGenerativeAIEmbeddings(
    model='models/embedding-001',
    google_api_key='your_api'
)

loader=PyPDFLoader("Bahurdeen-resume.pdf")
docs=loader.load()

chunk=RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

a=chunk.split_documents(docs)
vect=FAISS.from_documents(a,embedd)

model=ChatGoogleGenerativeAI(
    model='gemini-2.5-flash',
    google_api_key="your_api"
)

'''b=input("enter the question :")
con_vect=vect.similarity_search(b,k=5)
result = model.invoke("Based on these documents: " + "\n\n".join([d.page_content for d in con_vect]) + "\n\nAnswer the question: " + b)
print(result.content)'''
def chatbot_agentic_rag():
    print("Agentic RAG Chatbot is running! Type 'exit' to quit.")
    while True:
        user_query=input("You: ")
        if user_query.lower() == "exit":
            print("Chatbot session ended.")
            break
        convect = vect.similarity_search(user_query,k=5)
        context="\n".join([d.page_content for d in convect])
        prompt="you are an document based agent .for whatever questions asked ,give relevant most relavent answers from the document ,if the answer is not known in the given document then reply with a polite answer that you are not provided with the context.now here is the user question:{user_query},along with the context :{context} "
        try:
            response = model.invoke(prompt)
            print(f"Bot: {response.content}")
        except Exception as e:
            print(f"Error: {e}")

chatbot_agentic_rag()



