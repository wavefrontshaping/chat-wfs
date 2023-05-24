import os

import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS

from langchain.llms import OpenAI
from langchain import PromptTemplate

api_key = os.getenv("OPENAI_KEY")

MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.7
MAX_TOKENS = 1500


st.write()

with st.spinner("Loading data..."):
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local("faiss_index", embeddings)
    retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 5})

template = """
Use the following pieces of context to answer the scientific question at the end. 

{context}

Provide as much details as you can like providing latex equations rendered in markdown (use `$` around equations) when necessary about the following query:

{question}

Tell the user that they can find more information in the provided source list.

Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template,
)


default_query = "How to focus light through a scattering medium using the transmission matrix?"


def get_answer(chain, query):
    relevant_docs = retriever.get_relevant_documents(query)
    answer = chain.run(input_documents=relevant_docs, question=query)
    return answer, relevant_docs


# Tell me how to display a phase gradient on my SLM using Python
def app():
    # Streamlit app starts here
    st.title("Ask Wavefront Shaping")

    st.warning("""
        WARNING: Do not trust the results, this is experimental!
        (But you can trust the sources...)
    """
    )

    # model = st.selectbox("GPT Model", ("text-davinci-003", "gpt-3.5-turbo"))

    # temperature = st.slider("Temperature (higher is more creative)", 0.0, 1.0, (0.7))

    llm = OpenAI(model_name=MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, api_key=api_key)
    # chain = load_qa_chain(llm, chain_type="stuff")
    chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)

    # Text input for the user's question
    user_input = st.text_input("Ask a question", '', placeholder=default_query)

    if st.button("Submit Question"):
        with st.spinner("Wait for it..."):
            chat_result, docs = get_answer(chain, user_input)
        st.success("Done!")
        st.write(chat_result)

        with st.expander("See sources"):
            sources = {
                doc.metadata.get("Title", "")
                or doc.metadata.get("title", ""): doc.metadata.get("URL", "")
                or doc.metadata.get("source", "")
                for doc in docs
            }
            for title, url in sources.items():
                if title:
                    st.write(f"[{title}]({url})")


if __name__ == "__main__":
    app()

