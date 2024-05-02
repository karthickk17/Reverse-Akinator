from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_voyageai import VoyageAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import format_document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory


from streamlit_js_eval import streamlit_js_eval
import streamlit as st

from create_embeddings import generate_embeddings
import pandas as pd
import time
import random
import ast

#The center layout doesn't seem very nice to me
st.set_page_config(layout="wide")

st.title('Reverse Akinator (name subject to change)') #El Psy Congroo

#This variable is to ensure, the user doesn't change the difficult only the character is fixed.
if 'submit_flag' not in st.session_state:
    st.session_state['submit_flag'] = False

if 'names_list' not in st.session_state:
    st.session_state['names_list'] = []

if 'total_guesses' not in st.session_state:
    st.session_state['total_guesses'] = 5

if 'total_hints' not in st.session_state:
    st.session_state['total_hints'] = 3

if 'qa_pairs' not in st.session_state:
    st.session_state['qa_pairs'] = {}

if 'name' not in st.session_state:
    st.session_state['name'] = ""

if 'url' not in st.session_state:
    st.session_state['url'] = ""

if 'db' not in st.session_state:
    st.session_state['db'] = None

if 'hints' not in st.session_state:
    st.session_state['hints'] = []

if 'store' not in st.session_state:
    st.session_state['store'] = {}

#To add the question-answer to memory
def add_qa_pair(question, answer):
    st.session_state['qa_pairs'][question] = answer


#In easy mode, we allow the players to choose a specific set of sub group to play from
def easy_func():
    o1 = st.selectbox("Category", options=["Actor", "Cricketer"])
    if o1 == "Actor":
        o1_1 = st.selectbox("Language", options=["Tamil"])

    elif o1 == "Cricketer":
        o1_1 = st.selectbox("Region", options=["India"])

    print(o1, o1_1)

    df = pd.read_csv('data.csv')
    filtered_df = df[(df['Role'] == o1) & (df['Region'] == o1_1)]
    names_and_links = []
    for index, row in filtered_df.iterrows():
        names_and_links.append([row['Name'], row['wiki link']])
    st.session_state["names_list"] = names_and_links

#In hard mode, all the persons come to the pool.
def hard_func():
    df = pd.read_csv('data.csv')
    data_elements = [[row['Name'], row['wiki link']] for index, row in df.iterrows()]
    st.session_state["data_elements"] = data_elements

if not st.session_state['submit_flag']:
    diff = st.radio("Select Difficulty", ["Easy", "Hard"], horizontal=True)
    if diff == 'Easy':
        easy_func()    

    if diff == 'Hard':
        hard_func()

    st.session_state['submit_flag'] = st.button("Submit")
    if st.session_state['submit_flag']:
        st.info('The character is locked and loaded!')
        time.sleep(1)
        st.rerun()

with st.expander("""How to play?"""):
    st.markdown("""1. Choose difficulty: Easy or Hard. In easy mode, you can select the specific category of person while in hard mode, all the celebrities are included.
2. Once you have selected the character, you can start asking questions.
3. There are 3 hints available.
4. There is no limit to the number of questions that can be asked.
5. Enter your guess for the character's name in the guess text box and ask your questions at the bottom.""")

if not st.session_state['submit_flag']:
    st.stop()

#First column for image, second for AI response and third for previosuly asked questions
col1, col2, col3 = st.columns(3)

with col1:
    st.image('./assets/akinator happy 1.jpg', width=300)

#It stores the character's name
if st.session_state['name'] == "":
    res = random.choice(st.session_state["names_list"])
    res[0] = res[0].replace('.', '')
    st.session_state['name'] = res[0]

    st.session_state['url'] = res[1]
    st.session_state['db'] = generate_embeddings(st.session_state['url'].split('/')[-1])

# st.write(st.session_state['name'], st.session_state['url'])

#For this context voyage and openai embeddings both mxbai, nomic and snowflake-artic embeddings from ollama library
# st.session_state['db'] = Chroma(
#     persist_directory='./chroma-embeddings',
#     collection_name="rag-chroma-voyage",
#     embedding_function=VoyageAIEmbeddings(model="voyage-lite-02-instruct")
#     # embedding_function=OllamaEmbeddings(model="mxbai-embed-large")
#     # embedding_function=OpenAIEmbeddings()
# )

retriever = st.session_state['db'].as_retriever()

llm_hint = ChatGroq(temperature=0.2, model="mixtral-8x7b-32768")
llm_chat = ChatGroq(temperature=0.2, model="llama3-70b-8192", verbose=True, streaming=True)
# llm_chat = ChatOpenAI(model='gpt-3.5-turbo')
# llm_chat = ChatOllama(model='phi3:latest')

### Contextualize question ###
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm_chat, retriever, contextualize_q_prompt
)

qa_system_prompt = """You are a Game master who is a master at providing answers. Your task is answer the user's question in just three following ways:
    Yes
    No
    I don't know
    Don't answer otherwise.
    If the user asks question that is different from predicting the person in context, say Incorrect Question
    If the user asks question that cannot be answered using yes, no or I don't know, say Incorrect Question
    If the user tries to guess the name of the person in the question, say Incorrect Question
    Use I don't know if the current context doesn't provide the answer
    The user will use pronouns like he, she and words like your character which all refers to the character in context.
    Take alternative names seriously and must match the full spelling. If the user asks is his nickname thala but the context has thalapathy, then say no              
Answer the following question based only on the provided context:
<context>
{context}
</context>
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm_chat, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state['store']:
        st.session_state['store'][session_id] = ChatMessageHistory()
    return st.session_state['store'][session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

def combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    print(docs)
    return document_separator.join(doc_strings)

if st.session_state['hints'] == []:
    hint_prompt = ChatPromptTemplate.from_template(
    """You are a excellent Game Master! You are excellent at providing clues for the question! Generate the output in a list that is parsable. The output should contain only the list with three indices. Each index contains a clue. Help the user with his query:
    <context>
    {context}
    </context>
    {question}
    """
    )

    hint_chain = (
        {
            "context": retriever | combine_documents,
            "question": RunnablePassthrough()
        }
        | hint_prompt
        | llm_hint
        | StrOutputParser()
    )
    # st.session_state['name'] = 'Vijay'
    query = "Based on the given context, generate three important sentences and store in a list. If the sentences contain the name" + st.session_state['name'] + "replace it with X."
    msg0 = hint_chain.invoke(
            query
    )

    print(msg0)

    st.session_state['hints'] = ast.literal_eval(msg0)

prompt = st.chat_input("Enter your question", key="question")

with col2:
    ans_cont  = st.container()
    hint_cont  = st.container()

if prompt:
    msg1_result = conversational_rag_chain.invoke(
    {"input": prompt},
    config={
        "configurable": {"session_id": "abc123"}
    },  # constructs a key "abc123" in `store`.
    )
    print(msg1_result)
    msg1 = msg1_result["answer"]
    add_qa_pair(prompt, msg1)
    with ans_cont:
        st.chat_message("user").markdown(prompt)
        st.chat_message("assistant").markdown(msg1)

with hint_cont:
    guess = st.chat_input("Enter your guess here", key="guess_inp")
    hint_button = st.button("Hint")
    if hint_button and st.session_state['total_hints'] != -1:
        if st.session_state['total_hints'] == 0:
            st.warning('No hints available')
        else:
            st.session_state['total_hints'] -= 1
    for i in range(3-st.session_state['total_hints']):
            with st.expander(f"Hint {i+1}: "):
                st.write(st.session_state['hints'][i])

if guess:
    if guess.lower().strip() == st.session_state['name'].lower().strip():
        st.info("Congratulations! You Won!")
        time.sleep(3)
        st.info('The game will now reload!') 
        streamlit_js_eval(js_expressions="parent.window.location.reload()") 
    else:
        st.info(f"Incorrect answer! You still have {st.session_state['total_guesses']-1} left")
        st.session_state['total_guesses'] -= 1
    if st.session_state['total_guesses'] == 0:
        st.warning(f"Game Over! The answer is {st.session_state['name']}")
        time.sleep(3)
        st.info('The game will now reload!') 
        streamlit_js_eval(js_expressions="parent.window.location.reload()") 

with col3:
    st.write(f"Questions Asked: {len(st.session_state['qa_pairs'])}")
    for question, answer in st.session_state['qa_pairs'].items():
        with st.expander(f"Q: {question}"):
            st.write(f"A: {answer}")
