# %%
import gradio as gr
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, set_seed
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import fitz  
from langchain.agents import initialize_agent, load_tools

# %%
# Load environment variables.
load_dotenv()

# Set the model name for our LLMs.
OPENAI_MODEL = "gpt-3.5-turbo"
# Store the API key in a variable.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# %%
llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL, temperature=0.3)

# %%
# Create a Gradio Blocks instance for structuring the interface
app = gr.Blocks(fill_height=True)

#ChatBot function
def respond(message, chat_history):
    # Set up the built-in wikipedia tool.
    tools = load_tools(['wikipedia'], llm=llm)

    # Initialize the agent.
    agent = initialize_agent(tools, agent="chat-zero-shot-react-description", handle_parsing_errors=True ,llm=llm)
    bot_message = agent.run(message)
    chat_history.append((message, bot_message))
    return "", chat_history


def extract_text_from_pdf(file_data):
    text = ""
    try:
        print("Attempting to open PDF document...")
        # Open the PDF file from the uploaded file data
        pdf_document = fitz.open(stream=file_data, filetype="pdf")

        # Iterate over each page in the PDF
        print(f"Total number of pages: {pdf_document.page_count}")
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text += page.get_text()

        # Close the PDF document
        pdf_document.close()

    except Exception as e:
        print(f"Error: {e}")
    
    print(f"Extracted text length: {len(text)}")
    return text

#Forecast bot function
def pdfBot(query, file, file_data):
    text = ""
    try:
        print("Attempting to open PDF document...")
        # Open the PDF file from the uploaded file data
        pdf_document = fitz.open(stream=file_data, filetype="pdf")

        # Iterate over each page in the PDF
        print(f"Total number of pages: {pdf_document.page_count}")
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text += page.get_text()

        # Close the PDF document
        pdf_document.close()

    except Exception as e:
        print(f"Error: {e}")

    pdf_text = extract_text_from_pdf(file)

    format_template = f"{pdf_text}\n\n{query}"

    print(format_template)

    input_variables = ["query"]

    prompt_template = PromptTemplate(
        input_variables=input_variables,
        template=format_template
    )

    chain = LLMChain(llm=llm, prompt=prompt_template)

    result = chain.invoke({"query": query})

    return result["text"]


# Add tabs to the Gradio Blocks
with app:
    #A chatbot that will be used to talk with our first model we trained
    with gr.Tab("ChatBot"):
        chatbot = gr.Chatbot()
        msg = gr.Textbox(placeholder="Ask Your Question", show_label=False, scale=1)
        clear = gr.ClearButton([msg, chatbot])
        msg.submit(respond, [msg, chatbot], [msg, chatbot])


    with gr.Tab("PDFUploader"):
        gr.File(label="Upload a file:", type='binary'),
        
    #A basic input/output which will take some text and output a prediction that will be used with our second Model
    with gr.Tab("PDF Bot"):
        with gr.Row():

        
            gr.Interface(fn=pdfBot, allow_flagging='never',
                            inputs=[
                            gr.Textbox(lines=1, placeholder="Input", label="Question"),
                            gr.File(label="Upload a file:", type='binary'),
                            #gr.CheckboxGroup(["USA", "Japan", "Canada"], label="Countries", info="Where is your prediction"),
                            ],
                            outputs=gr.Textbox(lines=12,label="Results", show_copy_button=True, interactive=False))


# Launch the app
app.launch()


