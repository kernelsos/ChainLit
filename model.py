from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA, LLMChain
import chainlit as cl

# --- Configuration ---
DB_FAISS_PATH = 'vectorstores/db_faiss'

# --- Custom Prompt for Retrieval ---
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know.

Context: {context}
Question: {question}

Helpful answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

# --- Retrieval QA Chain ---
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

# --- Simple Chat Chain (Fast Mode) ---
def simple_chat_chain(llm):
    chat_prompt = PromptTemplate(
        input_variables=["question"],
        template="Answer conversationally and concisely.\n\nQuestion: {question}\nAnswer:"
    )
    return LLMChain(prompt=chat_prompt, llm=llm)

# --- Load Local Model ---
def load_llm():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

# --- Initialize Both Chains (Chat + Retrieval) ---
def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()

    qa_prompt = set_custom_prompt()
    retrieval_chain = retrieval_qa_chain(llm, qa_prompt, db)
    chat_chain = simple_chat_chain(llm)

    return {"chat": chat_chain, "retrieval": retrieval_chain}

# --- Chainlit Logic ---
@cl.on_message
async def main(message: cl.Message):
    # Load or reuse model chains
    chains = cl.user_session.get("chain")
    if not chains:
        chains = qa_bot()
        cl.user_session.set("chain", chains)

    user_query = message.content.lower()

    # --- Check if Retrieval Mode is Needed ---
    retrieval_triggers = ["source", "document", "reference", "knowledge", "pdf"]
    use_retrieval = any(keyword in user_query for keyword in retrieval_triggers)

    chain = chains["retrieval"] if use_retrieval else chains["chat"]

    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True

    # --- Run Selected Chain ---
    res = await chain.ainvoke({"question": message.content}, callbacks=[cb])

    # --- Format Response ---
    if use_retrieval:
        answer = res.get("result", "")
        sources = res.get("source_documents", [])
        formatted_sources = [doc.metadata.get("source", "Unknown").split("\\")[-1] for doc in sources]
        response_content = f"**Answer:**\n\n{answer}\n\n**Sources:**\n\n"
        response_content += "\n".join(f"- {src}" for src in formatted_sources) if formatted_sources else "No sources found."
    else:
        answer = res.get("text") or res.get("result", "")
        response_content = f"**Answer:**\n\n{answer}"

    await cl.Message(content=response_content).send()
