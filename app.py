from langchain_community.llms import Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import AIMessage, HumanMessage, SystemMessage  # Correct imports for messages
import time

from langchain_ollama import OllamaLLM

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the question asked."),
        ("user", "Question:{question}")
    ]
)

# Streamlit framework
st.title("Langchain Demo With Gemma Model")
input_text = st.text_input("What question you have in mind?")

# Ollama Llama2 model
llm = OllamaLLM(model="llama3")
output_parser = StrOutputParser()

if input_text:
    # Format input correctly for the LLM
    formatted_input = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=f"Question: {input_text}")
    ]
    
    # Stream the response token by token
    for token in llm.stream(formatted_input):
        st.write(token)
        time.sleep(0.3)  # Adjust the speed of token display
