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
from pathlib import Path
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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
# Set the model's file path
file_path = Path("./Models/emission.h5")

# Load the model to a new object
model = tf.keras.models.load_model(file_path)

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



# Define a function to make predictions
def make_predictions(year, emission, capTrade, eTarget, energy):
    try:
        # Apply value mapping for capTrade
        if capTrade == "Yes":
            capTrade_value = 1
        elif capTrade == "No":
            capTrade_value = 0
        else:
            raise ValueError("capTrade must be either 'Yes' or 'No'")

        # Create a dataframe with 5 columns containing the input number
        df = pd.DataFrame([year, emission, capTrade_value, eTarget, energy])

        scaler = StandardScaler()
        scaler.fit(df)
        df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
        
        # Convert the dataframe to a numpy array and reshape it to match the model input shape
        input_array = df_scaled.values.reshape(1, -1)  # Reshape based on dataframe shape
        
        # Generate predictions using the loaded model
        predictions = model.predict(input_array, verbose=0)
        
        return f"{predictions[0][0]:.2f}°F"

    except ValueError as ve:
        return f"ValueError: {ve}"
    
    except Exception as e:
        return f"Error: {e}"



    # Function to preprocess the file
def predict_data(input_file):
    # Load the data from the input file
    df = pd.read_csv(input_file)
    #Drop country from data frame
    df.drop(columns='Country', axis=1,inplace=True)
    
    scaler = StandardScaler()
    scaler.fit(df)
    df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)

    # Make predictions using the loaded Keras model
    predictions = model.predict(df_scaled)

    #append predicted values
    df['temp_change'] = predictions

    mean_temp_change =df.groupby('Year')['temp_change'].mean()

    # Plotting the data
    plt.figure(figsize=(10, 8))
    plt.plot(mean_temp_change.index, mean_temp_change.values, marker='o')
    plt.xlabel('Year')
    plt.ylabel('Mean Temperature Change')
    plt.title('Mean Temperature Change by Year')
    plt.grid(True)
    plt.xticks(mean_temp_change.index)
    
    plt.savefig('output_chart.png')

    return 'output_chart.png'



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
        msg = gr.Textbox(placeholder="Ask Your Question", show_label=False, scale=1, value="What is Cap and Trade?")

        # Create a ClearButton component that clears the textbox and chatbot when clicked.
        clear = gr.ClearButton([msg, chatbot])

        # Submit user input from the textbox to the chatbot when submitted.
        msg.submit(chatbotfunc, [msg, chatbot], [msg, chatbot])


    with gr.Tab("PredictionBot"):
        years = list(range(2025, 2040))
        gr.Interface(
            fn=make_predictions,  # Use the prediction function as the function to be executed
            allow_flagging='never',
            inputs = [
                    gr.Dropdown(choices=years, label="Select Year", value=2026),  # Input component for specifying number of predictions
                    gr.Number(label="Current Emissions (Metric Ton)", value=36729217),  # Input component for specifying number of predictions
                    gr.Radio(choices=["Yes", "No"], label="Choose an option", value="Yes"), # Input component for specifying number of predictions
                    gr.Number(label="Emission Target", value=472),  # Input component for specifying number of predictions
                    gr.Number(label="Current Renewable Energy", value=1519),  # Input component for specifying number of predictions
            ],
            outputs=gr.Textbox(label="Temperature Prediction", show_copy_button=True, interactive=False),  # Output component to display model predictions
            title="Model Predictions",  # Title for the interface
            description="Enter the values you need in their corresponding inputs.",  # Description for the interface
        )

    with gr.Tab("VisualBot"):
        gr.Interface(fn=predict_data, 
                     inputs=gr.File(label="Upload Data File"), 
                     outputs="image",
                     allow_flagging=False)

    # Create a tab for the PDF Bot.
    with gr.Tab("PDFBot"):
            # Define the Gradio interface for the PDF Bot.
            gr.Interface(fn=pdfBot, title="Upload Your Desired PDF", allow_flagging='never',
                        inputs=[
                            gr.Textbox(lines=1, placeholder="Input", label="Ask a Question about your PDF:", value="What is this PDF about?"),
                            gr.File(label="12 pages or less", type='binary'),
                        ],
                        outputs=gr.Textbox(lines=16, label="Answer:", show_copy_button=True, interactive=False))



# Launch the app
app.launch()



