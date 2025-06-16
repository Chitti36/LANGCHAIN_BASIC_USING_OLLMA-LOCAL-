import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama  # ✅ FIXED
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# LangSmith tracking (optional)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{Question}")
])

# Streamlit UI
st.title("LangChain Demo using Gemma Model")
input_text = st.text_input("What's in your mind?")

# Load Ollama model
llm = Ollama(model="gemma:2b")  # ✅ FIXED
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Handle input
if input_text:
    response = chain.invoke({"Question": input_text})
    st.write(response)
