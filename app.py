import gradio as gr
import os
import torch
from medical_chatbot import ColabBioGPTChatbot

def initialize_chatbot():
    """Initialize the chatbot with proper error handling"""
    try:
        print("🚀 Initializing Medical Chatbot...")
        
        # Check if GPU is available but use CPU for stability on HF Spaces
        use_gpu = torch.cuda.is_available()
        use_8bit = use_gpu  # Only use 8-bit if GPU is available
        
        chatbot = ColabBioGPTChatbot(use_gpu=use_gpu, use_8bit=use_8bit)
        
        # Try to load medical data
        medical_file = "Pediatric_cleaned.txt"
        if os.path.exists(medical_file):
            chatbot.load_medical_data(medical_file)
            status = f"✅ Medical file '{medical_file}' loaded successfully! Ready to chat!"
            success = True
        else:
            status = f"❌ Medical file '{medical_file}' not found. Please ensure the file is in the same directory."
            success = False
            
        return chatbot, status, success
        
    except Exception as e:
        error_msg = f"❌ Failed to initialize chatbot: {str(e)}"
        print(error_msg)
        return None, error_msg, False

# Debug file check
medical_file = "Pediatric_cleaned.txt"
print(f"Debug: Looking for file: {medical_file}")
print(f"Debug: File exists: {os.path.exists(medical_file)}")
if os.path.exists(medical_file):
    with open(medical_file, 'r') as f:
        content = f.read()
    print(f"Debug: File size: {len(content)} characters")

# Initialize chatbot at startup
print("🏥 Starting Pediatric Medical Assistant...")
chatbot, startup_status, medical_file_loaded = initialize_chatbot()

# Debug information
print(f"Debug: Medical file loaded = {medical_file_loaded}")
if chatbot and hasattr(chatbot, 'knowledge_chunks'):
    print(f"Debug: Number of knowledge chunks = {len(chatbot.knowledge_chunks)}")
    if chatbot.knowledge_chunks:
        print(f"Debug: First chunk preview = {chatbot.knowledge_chunks[0]['text'][:100]}...")
else:
    print("Debug: No knowledge_chunks attribute found")

def generate_response(user_input, history):
    """Generate response with proper error handling"""
    if not chatbot:
        return history + [("System Error", "❌ Chatbot failed to initialize. Please refresh the page and try again.")], ""
    
    if not medical_file_loaded:
        return history + [(user_input, "⚠️ Medical data failed to load. The chatbot may not have access to the full medical knowledge base.")], ""
    
    if not user_input.strip():
        return history, ""
    
    try:
        # Generate response
        bot_response = chatbot.chat(user_input)
        
        # Add to history
        history = history + [(user_input, bot_response)]
        
        return history, ""
        
    except Exception as e:
        error_response = f"⚠️ Sorry, I encountered an error: {str(e)}. Please try rephrasing your question."
        history = history + [(user_input, error_response)]
        return history, ""

# Create custom CSS for better styling
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.chatbot {
    height: 500px !important;
}

.message {
    padding: 10px;
    margin: 5px;
    border-radius: 10px;
}

.user-message {
    background-color: #e3f2fd;
    margin-left: 20%;
}

.bot-message {
    background-color: #f5f5f5;
    margin-right: 20%;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, title="Pediatric Medical Assistant") as demo:
    gr.Markdown(
        """
        # 🩺 Pediatric Medical Assistant
        
        Welcome to your AI-powered pediatric medical assistant! This chatbot uses advanced medical AI (BioGPT) 
        to provide evidence-based information about children's health and medical conditions.
        
        **⚠️ Important Disclaimer:** This tool provides educational information only. 
        Always consult qualified healthcare professionals for medical diagnosis, treatment, and personalized advice.
        """
    )
    
    # Display startup status
    gr.Markdown(f"**System Status:** {startup_status}")
    
    # Chat interface
    with gr.Row():
        with gr.Column(scale=4):
            chatbot_ui = gr.Chatbot(
                label="💬 Chat with Medical AI",
                height=500,
                show_label=True,
                avatar_images=("👤", "🤖")
            )
            
            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="Ask a pediatric health question... (e.g., 'What causes fever in children?')",
                    lines=2,
                    max_lines=5,
                    show_label=False,
                    scale=4
                )
                submit_btn = gr.Button("Send 📤", variant="primary", scale=1)
        
        with gr.Column(scale=1):
            gr.Markdown(
                """
                ### 💡 Example Questions:
                
                - "What causes fever in children?"
                - "How to treat a child's cough?"
                - "When should I call the doctor?"
                - "What are signs of dehydration?"
                - "How to prevent common infections?"
                - "What are normal vital signs for kids?"
                - "How to manage childhood allergies?"
                
                ### 🔧 System Info:
                - **Model:** BioGPT (Medical AI)
                - **Specialization:** Pediatric Medicine
                - **Search:** Vector + Keyword Hybrid
                - **Knowledge Base:** Pediatric Medical Data
                """
            )
    
    # Event handlers
    def submit_message(user_msg, history):
        return generate_response(user_msg, history)
    
    # Connect events
    user_input.submit(
        fn=submit_message,
        inputs=[user_input, chatbot_ui],
        outputs=[chatbot_ui, user_input],
        show_progress=True
    )
    
    submit_btn.click(
        fn=submit_message,
        inputs=[user_input, chatbot_ui],
        outputs=[chatbot_ui, user_input],
        show_progress=True
    )
    
    # Footer
    gr.Markdown(
        """
        ---
        **🏥 Medical AI Assistant** | Powered by BioGPT | For Educational Purposes Only
        
        **Remember:** Always consult healthcare professionals for medical emergencies and personalized medical advice.
        
        **Technical Note:** This system uses semantic search to find relevant medical information from a curated pediatric database, 
        then generates contextual responses using BioGPT, a medical AI model trained on biomedical literature.
        """
    )

# Launch configuration for Hugging Face Spaces
if __name__ == "__main__":
    # For Hugging Face Spaces deployment
    demo.launch(
        server_name="0.0.0.0",  # Required for HF Spaces
        server_port=7860,       # Default port for HF Spaces
        show_error=True,        # Show errors for debugging
        share=False             # Don't create public links in HF Spaces
    )