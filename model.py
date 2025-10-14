from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
import time

# =====================================================
# CONFIG
# =====================================================
DB_FAISS_PATH = "vectorstores/db_faiss"
LLAMA_MODEL_PATH = "./llama-2-7b-chat.ggmlv3.q8_0.bin"

custom_prompt_template = """Use the following context to answer the user's question.
If you don't know the answer, just say you don't know. Do not make up an answer.

Context:
{context}

Question:
{query}

Helpful Answer:
"""


def set_custom_prompt():
    """Return custom prompt template for RetrievalQA."""
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "query"],
    )


def load_llm():
    """Load the local LLaMA model using CTransformers."""
    return CTransformers(
        model=LLAMA_MODEL_PATH,
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5,
    )


def qa_bot():
    """Initialize QA chain with FAISS and LLM."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    prompt = set_custom_prompt()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return qa


def final_result(query):
    """Run a one-off query without Chainlit."""
    qa = qa_bot()
    response = qa({"query": query})
    return response


# =====================================================
# CHAINLIT APP
# =====================================================
@cl.on_chat_start
async def start():
    """Initialize chatbot session."""
    time.sleep(10)
    msg = cl.Message(content="Starting the Medical Assistant bot...")
    await msg.send()

    try:
        chain = qa_bot()
        msg.content = "👋 Hi! I'm your Medical Assistant. How can I help you today?"
        await msg.update()
        cl.user_session.set("chain", chain)
    except Exception as e:
        msg.content = f"❌ Error initializing bot: {str(e)}\nPlease run 'python ingest.py' first."
        await msg.update()


@cl.on_message
async def main(message: cl.Message):
    """Handle user messages."""
    chain = cl.user_session.get("chain")
    if chain is None:
        await cl.Message("Bot not initialized. Please refresh the page.").send()
        return

    try:
        res = await chain.ainvoke(message.content)
        answer = res["result"]
        sources = res.get("source_documents", [])

        if sources:
            answer += "\n\n📚 **Sources:**"
            for i, doc in enumerate(sources, 1):
                src = doc.metadata.get("source", "Unknown")
                pg = doc.metadata.get("page", "N/A")
                answer += f"\n{i}. {src} (Page {pg})"

        await cl.Message(content=answer).send()

    except Exception as e:
        await cl.Message(content=f"⚠️ Error: {str(e)}").send()
