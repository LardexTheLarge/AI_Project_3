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
def chatbotfunc(message, chat_history):
    # Set up the built-in Wikipedia tool.
    tools = load_tools(['wikipedia'], llm=llm)

    # Initialize the agent with specified parameters.
    # 'agent' parameter specifies the type of agent to use ('chat-zero-shot-react-description').
    # 'handle_parsing_errors' parameter is set to True to handle any parsing errors that may occur.
    # 'llm' parameter is passed to the agent for language model configuration.
    agent = initialize_agent(tools, agent="chat-zero-shot-react-description", handle_parsing_errors=True, llm=llm)

    # Run the agent to generate a response based on the input message.
    bot_message = agent.run(message)

    # Append the message and bot's response to the chat history.
    chat_history.append((message, bot_message))

    # Return an empty string (response) along with the updated chat history.
    return "", chat_history



def pdfBot(query, file_data):
    # Check if file_data is None (no file uploaded)
    if file_data is None:
        return "Error: No PDF file uploaded."

    text = ""
    try:
        # Open the PDF file
        pdf_document = fitz.open(stream=file_data, filetype="pdf")

        # Iterate over each page in the PDF
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text += page.get_text()

        # Close the PDF document
        pdf_document.close()

    except Exception as e:
        return f"Error: {e}"

    # Check if no text was extracted
    if not text:
        return "Error: No text extracted from the PDF."

    # Construct the format template
    format_template = f"{text}\n\n{query}"

    # Define input variables for the prompt template
    input_variables = ["query"]

    # Create a prompt template
    prompt_template = PromptTemplate(
        input_variables=input_variables,
        template=format_template
    )

    # Create an LLMChain instance
    chain = LLMChain(llm=llm, prompt=prompt_template)

    # Invoke the chain with the query
    result = chain.invoke({"query": query})

    return result["text"]


# Define the interface using Gradio.
with app:
    # Create a tab for the ChatBot.
    with gr.Tab("ChatBot"):
        # Initialize a Chatbot component.
        chatbot = gr.Chatbot()

        # Create a textbox for user input without a label and scale it to size 1.
        msg = gr.Textbox(placeholder="Ask Your Question", show_label=False, scale=1)

        # Create a ClearButton component that clears the textbox and chatbot when clicked.
        clear = gr.ClearButton([msg, chatbot])

        # Submit user input from the textbox to the chatbot when submitted.
        msg.submit(chatbotfunc, [msg, chatbot], [msg, chatbot])

    # Create a tab for the PDF Bot.
    with gr.Tab("PDF Bot"):
            # Define the Gradio interface for the PDF Bot.
            gr.Interface(fn=pdfBot, title="Upload Your Desired PDF", allow_flagging='never',
                        inputs=[
                            gr.Textbox(lines=1, placeholder="Input", label="Ask a Question about your PDF:"),
                            gr.File(label="12 pages or less", type='binary'),
                        ],
                        outputs=gr.Textbox(lines=16, label="Answer:", show_copy_button=True, interactive=False))



# Launch the app
app.launch()


