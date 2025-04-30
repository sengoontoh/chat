import base64
import io
import os
import streamlit as st
import time # Needed for unique key generation
from langfuse import Langfuse  # Add Langfuse import
from langfuse.openai import openai
import requests  # For API calls

# Default prompt to use as fallback
DEFAULT_SYSTEM_PROMPT = "You are a useless assistant."

# Initialize Langfuse client
try:
    langfuse = Langfuse(
        public_key=st.secrets.get("LANGFUSE_PUBLIC_KEY"),
        secret_key=st.secrets.get("LANGFUSE_SECRET_KEY"),
        host=st.secrets.get("LANGFUSE_HOST", "https://cloud.langfuse.com")  # Default to Langfuse Cloud
    )
except Exception as e:
    langfuse = None
    st.warning(f"Langfuse initialization failed: {e}. Continuing without analytics.")

# --- Callback Function ---
# This function will be called whenever the file uploader changes.
def handle_file_upload():
    # This flag tells us the uploader state changed and might need processing
    if st.session_state.get(st.session_state.uploader_key) is not None:
         st.session_state.process_new_upload = True
    else:
         st.session_state.process_new_upload = False

# Function to get system prompt from Langfuse or use default
def get_system_prompt():
    try:
        if langfuse:
            # Get the prompt object
            prompt_obj = langfuse.get_prompt("main")
            # Extract the text - use appropriate method depending on Langfuse version
            return prompt_obj.prompt  # or prompt_obj.text or prompt_obj.content
        else:
            return DEFAULT_SYSTEM_PROMPT
    except Exception as e:
        st.warning(f"Error getting prompt from Langfuse: {e}. Using default prompt.")
        return DEFAULT_SYSTEM_PROMPT

st.title("Chat")

# OpenAI API parameters
OPENAI_MODEL = "chatgpt-4o-latest"
TEMPERATURE = 1.0  # Default temperature (0.0 to 2.0)
TOP_P = 1.0        # Default top_p (0.0 to 1.0)

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# --- Initialize Session State ---
if "openai_model" not in st.session_state:
    # Ensure the model supports vision, like gpt-4o or gpt-4-turbo
    st.session_state["openai_model"] = OPENAI_MODEL
if "messages" not in st.session_state:
    st.session_state.messages = []
# Initialize our flag for tracking if the uploaded file should be processed
if "process_new_upload" not in st.session_state:
    st.session_state.process_new_upload = False
# Initialize the dynamic key for the file uploader
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = "file_uploader_0"
# Initialize unique user ID for Langfuse traces
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{int(time.time())}"
# Initialize session trace ID for Langfuse
if "trace_id" not in st.session_state:
    st.session_state.trace_id = f"session_{int(time.time())}"
# Store system prompt in session state
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = get_system_prompt()

# --- Initiate Conversation on First Load ---
if not st.session_state.messages:
    try:
        # Use system prompt from session state
        initial_api_messages = [{"role": "system", "content": st.session_state.system_prompt}]
        
        # Standard OpenAI call
        completion = openai.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=initial_api_messages,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        
        initial_response = completion.choices[0].message.content
        
        # Add the initial assistant message to the history
        st.session_state.messages.append({"role": "assistant", "content": initial_response})
    except Exception as e:
        st.error(f"Failed to initialize conversation: {e}")
        # Add a fallback message if the API call fails
        st.session_state.messages.append({"role": "assistant", "content": "Hello! I couldn't connect properly, but I'm here to help."})

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Check if content is a list (multimodal) or string (text only)
        if isinstance(message["content"], list):
            for item in message["content"]:
                if item["type"] == "text":
                    st.markdown(item["text"])
                elif item["type"] == "image_url":
                    # Extract base64 string and display image
                    try:
                        base64_image = item["image_url"]["url"].split(",")[1]
                        image_bytes = base64.b64decode(base64_image)
                        st.image(image_bytes)
                    except (IndexError, ValueError) as e:
                        st.error(f"Error displaying image: {e}")
        else:
            st.markdown(message["content"]) # Original text-only messages

# Chat input for text prompt
if prompt := st.chat_input("How can I help you today?"):    
    user_message_content = []
    image_bytes_for_display = None # Variable to hold image bytes for immediate display
    image_was_processed_this_run = False # Track if we used the image

    # Check if a file exists *AND* if our flag indicates it's a new/unprocessed upload
    current_file_state = st.session_state.get(st.session_state.uploader_key) # Use state via key
    if current_file_state is not None and st.session_state.get('process_new_upload', False):
        # Read image bytes and encode to base64
        bytes_data = current_file_state.getvalue()
        base64_string = base64.b64encode(bytes_data).decode('utf-8')
        
        # Determine mime type
        mime_type = current_file_state.type or "image/jpeg" 

        # Add image data (as base64 URL) to the message content list for the API
        user_message_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_string}"
            }
        })
        # Store the raw bytes just for displaying in the *current* user message block
        image_bytes_for_display = bytes_data 
        
        # --- Crucial: Reset the flag after processing ---
        # This prevents the same upload from being processed again on the next message
        st.session_state.process_new_upload = False 
        image_was_processed_this_run = True

    # Add text prompt to the message content list
    user_message_content.append({"type": "text", "text": prompt})
    
    # Add the complete user message (list with text and potentially image URL) 
    # to the session state history BEFORE displaying it.
    st.session_state.messages.append({"role": "user", "content": user_message_content})

    # Display the user's submitted message in the chat interface
    with st.chat_message("user"):
        # Display the image *if* one was processed for this specific submission
        if image_bytes_for_display:
            st.image(image_bytes_for_display)
        # Display the text prompt
        st.markdown(prompt)

    # Get assistant response
    with st.chat_message("assistant"):
        try:
            # Prepare messages for API, starting with the system prompt from session state
            api_messages = [
                {"role": "system", "content": st.session_state.system_prompt} 
            ]
            # Add the rest of the conversation history
            for msg in st.session_state.messages:
                content = msg["content"]
                # Ensure user message content is always a list for the API
                if isinstance(content, str) and msg["role"] == "user":
                     content = [{"type": "text", "text": content}] 
                # Append assistant messages as strings, user messages as lists
                api_messages.append({"role": msg["role"], "content": content})

            # Standard OpenAI call without Langfuse wrapper
            stream = openai.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=api_messages,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                stream=True,
            )
                
            response = st.write_stream(stream)
            
            # Add only the assistant's response to the display history
            st.session_state.messages.append({"role": "assistant", "content": response})
                            
        except Exception as e:
            st.error(f"An error occurred while contacting the API: {e}")
            # Log the error to Langfuse
            if langfuse:
                trace = langfuse.trace(
                    id=st.session_state.trace_id,
                    user_id=st.session_state.user_id,
                )
                trace.event(
                    name="api_error",
                    input=str(e),
                    level="ERROR"
                )

    # --- Reset Uploader Widget via Key Change ---
    # If we processed an image in this run, change the key and rerun
    if image_was_processed_this_run:
        # Generate a new unique key using timestamp
        st.session_state.uploader_key = f"file_uploader_{int(time.time())}"
        
        # Rerun the script immediately
        st.rerun() 