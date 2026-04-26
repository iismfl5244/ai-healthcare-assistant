import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from datetime import datetime, timedelta
import random
import tempfile
import pandas as pd
import os


# -----------------------------
# Streamlit setup
# -----------------------------
st.set_page_config(
    page_title="Mike Leikam - Capstone - AI Healthcare Assistant", layout="wide"
)

st.title("Mike Leikam - Capstone - AI Healthcare Assistant")
st.write(
    "Upload a patient report, ask report questions, search PubMed, or book an appointment."
)

st.warning("This tool is for educational purposes only. It is not medical advice.")


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Assistant Modules")
st.sidebar.write("📄 Patient Report RAG")
st.sidebar.write("🔬 PubMed Research")
st.sidebar.write("📅 Appointment Booking")
st.sidebar.write("🧠 Session Memory")
st.sidebar.write("📊 Excel Interaction Log")


# -----------------------------
# Models and tools
# -----------------------------
llm = ChatOllama(model="llama3.2:latest", temperature=0.1)
pubmed = PubmedQueryRun()


# -----------------------------
# Session state
# -----------------------------
if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "memory" not in st.session_state:
    st.session_state.memory = {"patient_name": None, "last_appointment": None}

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Create retriever from uploaded PDF
# -----------------------------
def create_retriever(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    loader = PyPDFLoader(temp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    chunks = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever()


# -----------------------------
# Answer from patient report
# -----------------------------
def answer_from_report(question):
    docs = st.session_state.retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a medical assistant.

Answer the user's question using ONLY the patient report context below.

If the user asks to summarize the report, provide a concise summary of:
- diagnosis or chief complaint
- medications
- lab findings
- notable exam findings
- recommendations if present

If the answer is not in the report, say:
"I cannot find that information in the report."

Patient Report Context:
{context}

User Question:
{question}
"""

    response = llm.invoke(prompt)

    return response.content, context


# -----------------------------
# Search PubMed
# -----------------------------
def search_pubmed(question):
    results = pubmed.invoke(question)

    prompt = f"""
You are a medical assistant.

Summarize the PubMed research results below in simple language.
This is educational information, not medical advice.

Do not diagnose the patient.
Do not replace professional medical judgment.

PubMed Results:
{results}

User Question:
{question}
"""

    response = llm.invoke(prompt)
    return response.content


# -----------------------------
# Book appointment
# -----------------------------
def book_appointment(question):
    today = datetime.now()

    if "tomorrow" in question.lower():
        appointment_date = today + timedelta(days=1)
    else:
        appointment_date = today

    hour = 14

    if "3" in question:
        hour = 15
    elif "4" in question:
        hour = 16
    elif "5" in question:
        hour = 17

    appointment_date = appointment_date.replace(
        hour=hour, minute=0, second=0, microsecond=0
    )

    token = random.randint(100000, 999999)

    result = (
        f"✅ Appointment booked.\n\n"
        f"🕒 Time: {appointment_date.strftime('%Y-%m-%d %H:%M')}\n\n"
        f"🔢 Token: {token}"
    )

    st.session_state.memory["last_appointment"] = result
    return result


# -----------------------------
# Route user question
# -----------------------------
def route_question(question):
    q = question.lower()

    if (
        "report" in q
        or "patient" in q
        or "diagnosis" in q
        or "summarize" in q
        or "summary" in q
        or "medication" in q
        or "lab" in q
    ):
        return "REPORT"

    elif "latest" in q or "research" in q or "treatment" in q or "pubmed" in q:
        return "RESEARCH"

    elif "book" in q or "appointment" in q or "schedule" in q:
        return "BOOKING"

    elif "last appointment" in q or "patient name" in q or "memory" in q:
        return "MEMORY"

    else:
        return "UNKNOWN"


# -----------------------------
# Log interaction to Excel
# -----------------------------
def log_to_excel(question, response, route):
    file_name = "chat_log.xlsx"

    new_data = {
        "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "Question": [question],
        "Response": [response],
        "Route": [route],
    }

    df_new = pd.DataFrame(new_data)

    if os.path.exists(file_name):
        df_existing = pd.read_excel(file_name)
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_updated = df_new

    df_updated.to_excel(file_name, index=False)


# -----------------------------
# File upload
# -----------------------------
uploaded_file = st.file_uploader("Upload patient PDF report", type=["pdf"])

if uploaded_file is not None and st.session_state.retriever is None:
    with st.spinner("Processing PDF..."):
        st.session_state.retriever = create_retriever(uploaded_file)
    st.success("PDF processed and ready.")


# -----------------------------
# Display chat history
# -----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# -----------------------------
# Chat input
# -----------------------------
user_input = st.chat_input("Ask about the report, research, or booking")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.write(user_input)

    route = route_question(user_input)
    st.caption(f"Route selected: {route}")

    source_context = None

    if route == "REPORT":
        if st.session_state.retriever is None:
            response = "Please upload a PDF report first."
        else:
            response, source_context = answer_from_report(user_input)

    elif route == "RESEARCH":
        response = search_pubmed(user_input)

    elif route == "BOOKING":
        response = book_appointment(user_input)

    elif route == "MEMORY":
        if "last appointment" in user_input.lower():
            response = st.session_state.memory.get(
                "last_appointment", "No appointment has been booked yet."
            )
        elif "patient name" in user_input.lower():
            response = st.session_state.memory.get(
                "patient_name", "No patient name is stored in memory."
            )
        else:
            response = str(st.session_state.memory)

    else:
        response = "Please ask about the patient report, medical research, or booking an appointment."

    st.session_state.messages.append({"role": "assistant", "content": response})

    log_to_excel(user_input, response, route)

    with st.chat_message("assistant"):
        st.write(response)

        if source_context:
            with st.expander("Source context used"):
                st.write(source_context[:1500])


# -----------------------------
# Download Excel log
# -----------------------------
if os.path.exists("chat_log.xlsx"):
    with open("chat_log.xlsx", "rb") as file:
        st.sidebar.download_button(
            label="Download Excel Chat Log",
            data=file,
            file_name="chat_log.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
