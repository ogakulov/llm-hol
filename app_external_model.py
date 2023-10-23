import gradio as gr
import requests
import os
import re

# Model Access constants
cml_model_service = "https://modelservice.%s/model" % os.getenv('CDSW_DOMAIN')
llm_model_request_json_template = '{"accessKey":"%s","request":{"prompt":"%s"}}'
embedding_model_request_json_template = '{"accessKey":"%s","request":{"chunk":"%s"}}'

EMBEDDINGS_MODEL_ID ="m52yii0wuhzhs6l7001jm0s17m7lkolb"
LLAMA_CHAT_MODEL_ID = "m8rwie5pevcltiebfn9x74gyv2ajv649"



# LOCAL VECTOR DB SECTION

# In this section we spin upa vector db that makes use of on-disk collection info in milvus-data
# This should have been previouslty populated by running 2_populate_vector_db_job in a session or job 

from milvus import default_server
from pymilvus import connections, Collection, utility

# Start Milvus Vector DB
default_server.stop()
default_server.set_base_dir('milvus-data')
default_server.start()


try:
    connections.connect(alias='default', host='localhost', port=default_server.listen_port)   
except Exception as e:
    default_server.stop()
    raise e
    
print(utility.get_server_version())

def get_embedding_from_shared(text):
  
    data = embedding_model_request_json_template%(EMBEDDINGS_MODEL_ID, text.replace("\n", "\\n"))

    r = requests.post(cml_model_service, 
                      data=data, 
                      headers={'Content-Type': 'application/json'})
    print("Getting relevant chunk from vector db")
    embedding = prediction = r.json()["response"]
    return embedding
  
# This function extracts the most relevant chunk to the user question
def get_relevant_chunk(question):
    
    vector_db_collection = Collection('cloudera_ml_docs')
    vector_db_collection.load()
    
    # Generate embedding for user question
    question_embedding =  get_embedding_from_shared(question)
    
    # Define search attributes for Milvus vector DB
    vector_db_search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    
    # Execute search and get nearest vector, outputting the relativefilepath
    nearest_vectors = vector_db_collection.search(
        data=[question_embedding], # The data you are querying on
        anns_field="embedding", # Column in collection to search on
        param=vector_db_search_params,
        limit=1, # limit results to 1
        expr=None, 
        output_fields=['relativefilepath'], # The fields you want to retrieve from the search result.
        consistency_level="Strong"
    )
    
    # Print the file path of the kb chunk
    print(nearest_vectors[0].ids[0])
    
    # Return text of the nearest knowledgebase chunk
    kb_path = nearest_vectors[0].ids[0]
    
    with open(kb_path, "r") as f: # Open file in read mode
        return f.read()
    #return load_context_chunk_from_data(nearest_vectors[0].ids[0])



# MODEL ACCESS SECTION


model_access_key = LLAMA_CHAT_MODEL_ID




# Adapted from https://colab.research.google.com/drive/1SSv6lzX3Byu50PooYogmiwHqf5PQN68E?usp=sharing#scrollTo=PeEh17FDLzEe

    
SYSTEM_PROMPT = """<s>[INST] <<SYS>>
You are a helpful knowledgebase bot. Answer questions based on the context given. Your answers are clear and concise. Dont use emojis.
<</SYS>>

"""

# Formatting function for message and history
def format_message(message: str, history: list, memory_limit: int = 3) -> str:
    """
    Formats the message and history for the Llama model.

    Parameters:
        message (str): Current message to send.
        history (list): Past conversation history.
        memory_limit (int): Limit on how many past interactions to consider.

    Returns:
        str: Formatted message string
    """
    
    #get closest vector for this question/message
    knowledgebase_context = get_relevant_chunk(message)
    
    message_with_knowledge = "\n[context]: %s \n\n[question]: %s" % (knowledgebase_context , message)
    
    
    # always keep len(history) <= memory_limit
    if len(history) > memory_limit:
        history = history[-memory_limit:]

    if len(history) == 0:
        return SYSTEM_PROMPT + f"{message_with_knowledge} [/INST]"

    formatted_message = SYSTEM_PROMPT + f"{history[0][0]} [/INST] {history[0][1]} </s>"

    # Handle conversation history
    for user_msg, model_answer in history[1:]:
        formatted_message += f"<s>[INST] {user_msg} [/INST] {model_answer} </s>"
    print(formatted_message)
    # Handle the current message
    formatted_message += f"<s>[INST] {message_with_knowledge} [/INST]"
    print(formatted_message)

    return formatted_message

def llm_response(message, chat_history):
    """
    Calls out to CML Model for LLM responses
    """

    prompted_msg = format_message(message, chat_history)
    json_esc_prompted_msg = prompted_msg.replace("\n", "\\n")

    data = llm_model_request_json_template%(model_access_key,json_esc_prompted_msg)
    r = requests.post(cml_model_service,
                      data=data,
                      headers={'Content-Type': 'application/json'})

    prediction = r.json()["response"]
    print('--- Prompt sent to model')
    print(prompted_msg)
    print('--- Prediction returned from model')
    print(prediction)
    # bot_message is everything after the final [/INST] in the response
    start_of_bot = prediction.rfind('[/INST]')
    bot_message = prediction[start_of_bot+7:]

    return bot_message

  
  
##  Start Gradio UI  

demo = gr.ChatInterface(llm_response)
demo.launch(server_port=int(os.getenv('CDSW_APP_PORT')),
           enable_queue=True)


# Run the following in the session workbench to kill the gradio interface. Hit play button to rerun the script

# demo.close()