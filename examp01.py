
from langchain_core.runnables.utils import Output
import streamlit as st
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.chains import conversationBufferMemory

from constants import openai_key
import os

os.environ["OPENAI_API_KEY"] = openai_key

st.title('Langchain Demo with openAi')
input_text = st.text_input('Enter your text here')

#prompt 1
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template=(f"tell me about {input_text}")
)

#memory
person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')


llm=OpenAI(temperature=0.8)
chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True,Output_key='person',memory=person_memory)

#prompt 2
second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template=(f"when was {chain.output_key} born")


)

chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True,Output_key='DOB',memory=dob_memory)

#prompt 3

third_input_prompt = PromptTemplate(
input_variables=['DOB'],
template=(f"mention 5 major events happened around {chain.output_key} in the world")
) 

chain3 = LLMChain(llm=llm, prompt=third_input_prompt,
                  verbose=True,Output_key='description',memory=descr_memory)




parent_chain = SequentialChain(
       chains=[chain,chain2,chain3],
       input_variables=['name'],Output_variables=['person','DOB ','description'], verbose=True)


# OPENAI LLMS
llm = OpenAI(temperature=0.8)

if input_text:
     st.write(parent_chain(('name':input_text))
              
      with st.expander('Person Name'): 
      st.info(person_memory.buffer)

      with st.expander('Major Events'): 
      st.info(descr_memory.buffer)
             
