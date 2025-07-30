# BioGPT Medical Chatbot with Gradio Interface

import gradio as gr
import torch
import warnings
import numpy as np
import faiss
import os
import re
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import json

# Install required packages if not already installed
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Installing required packages...")
    import subprocess
    import sys
    
    packages = [
        "transformers>=4.21.0",
        "torch>=1.12.0", 
        "sentence-transformers",
        "faiss-cpu",
        "accelerate",
        "bitsandbytes",
        "datasets",
        "numpy",
        "sacremoses"
    ]
    
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from sentence_transformers import SentenceTransformer

# Suppress warnings
warnings.filterwarnings('ignore')

class GradioBioGPTChatbot:
    def __init__(self, use_gpu=True, use_8bit=True):
        """Initialize BioGPT chatbot for Gradio deployment"""
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        self.use_8bit = use_8bit and torch.cuda.is_available()
        
        # Initialize components
        self.setup_embeddings()
        self.setup_faiss_index()
        self.setup_biogpt()
        
        # Conversation tracking
        self.conversation_history = []
        self.knowledge_chunks = []
        self.is_data_loaded = False
        
    def setup_embeddings(self):
        """Setup medical-optimized embeddings"""
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            self.use_embeddings = True
        except Exception as e:
            print(f"Embeddings setup failed: {e}")
            self.embedding_model = None
            self.embedding_dim = 384
            self.use_embeddings = False

    def setup_faiss_index(self):
        """Setup FAISS for vector search"""
        try:
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            self.faiss_ready = True
        except Exception as e:
            print(f"FAISS setup failed: {e}")
            self.faiss_index = None
            self.faiss_ready = False

    def setup_biogpt(self):
        """Setup BioGPT model with optimizations"""
        model_name = "microsoft/BioGPT-Large"
        
        try:
            # Setup quantization config for memory efficiency
            if self.use_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                )
            else:
                quantization_config = None

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )

            if self.device == "cuda" and quantization_config is None:
                self.model = self.model.to(self.device)
                
        except Exception as e:
            print(f"BioGPT loading failed: {e}. Using fallback model...")
            self.setup_fallback_model()

    def setup_fallback_model(self):
        """Setup fallback model if BioGPT fails"""
        try:
            fallback_model = "microsoft/DialoGPT-medium"
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            self.model = AutoModelForCausalLM.from_pretrained(fallback_model)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            if self.device == "cuda":
                self.model = self.model.to(self.device)
                
        except Exception as e:
            print(f"All models failed: {e}")
            self.model = None
            self.tokenizer = None

    def create_medical_chunks(self, text: str, chunk_size: int = 400) -> List[Dict]:
        """Create medically-optimized text chunks"""
        chunks = []
        
        # Split by medical sections first
        medical_sections = self.split_by_medical_sections(text)
        
        chunk_id = 0
        for section in medical_sections:
            if len(section.split()) > chunk_size:
                # Split large sections by sentences
                sentences = re.split(r'[.!?]+', section)
                current_chunk = ""
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    if len(current_chunk.split()) + len(sentence.split()) < chunk_size:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk.strip():
                            chunks.append({
                                'id': chunk_id,
                                'text': current_chunk.strip(),
                                'medical_focus': self.identify_medical_focus(current_chunk)
                            })
                            chunk_id += 1
                        current_chunk = sentence + ". "
                
                if current_chunk.strip():
                    chunks.append({
                        'id': chunk_id,
                        'text': current_chunk.strip(),
                        'medical_focus': self.identify_medical_focus(current_chunk)
                    })
                    chunk_id += 1
            else:
                chunks.append({
                    'id': chunk_id,
                    'text': section,
                    'medical_focus': self.identify_medical_focus(section)
                })
                chunk_id += 1
                
        return chunks

    def split_by_medical_sections(self, text: str) -> List[str]:
        """Split text by medical sections"""
        section_patterns = [
            r'\n\s*(?:SYMPTOMS?|TREATMENT|DIAGNOSIS|CAUSES?|PREVENTION|MANAGEMENT).*?\n',
            r'\n\s*\d+\.\s+',
            r'\n\n+'
        ]
        
        sections = [text]
        for pattern in section_patterns:
            new_sections = []
            for section in sections:
                splits = re.split(pattern, section, flags=re.IGNORECASE)
                new_sections.extend([s.strip() for s in splits if len(s.strip()) > 100])
            sections = new_sections
            
        return sections

    def identify_medical_focus(self, text: str) -> str:
        """Identify the medical focus of a text chunk"""
        text_lower = text.lower()
        
        categories = {
            'pediatric_symptoms': ['fever', 'cough', 'rash', 'vomiting', 'diarrhea'],
            'treatments': ['treatment', 'therapy', 'medication', 'antibiotics'],
            'diagnosis': ['diagnosis', 'diagnostic', 'symptoms', 'signs'],
            'emergency': ['emergency', 'urgent', 'serious', 'hospital'],
            'prevention': ['prevention', 'vaccine', 'immunization', 'avoid']
        }
        
        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
                
        return 'general_medical'

    def load_medical_data_from_file(self, file_path: str) -> Tuple[str, bool]:
        """Load medical data from uploaded file"""
        if not file_path or not os.path.exists(file_path):
            return "❌ No file uploaded or file not found.", False
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # Create chunks
            chunks = self.create_medical_chunks(text)
            self.knowledge_chunks = chunks
            
            # Generate embeddings if available
            if self.use_embeddings and self.embedding_model and self.faiss_ready:
                success = self.generate_embeddings_and_index(chunks)
                if success:
                    self.is_data_loaded = True
                    return f"✅ Medical data loaded successfully! {len(chunks)} chunks processed with vector search.", True
            
            self.is_data_loaded = True
            return f"✅ Medical data loaded successfully! {len(chunks)} chunks processed (keyword search mode).", True
            
        except Exception as e:
            return f"❌ Error loading file: {str(e)}", False

    def generate_embeddings_and_index(self, chunks: List[Dict]) -> bool:
        """Generate embeddings and add to FAISS index"""
        try:
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            self.faiss_index.add(np.array(embeddings))
            return True
        except Exception as e:
            print(f"Embedding generation failed: {e}")
            return False

    def retrieve_medical_context(self, query: str, n_results: int = 3) -> List[str]:
        """Retrieve relevant medical context"""
        if self.use_embeddings and self.embedding_model and self.faiss_ready:
            try:
                query_embedding = self.embedding_model.encode([query])
                distances, indices = self.faiss_index.search(np.array(query_embedding), n_results)
                context_chunks = [self.knowledge_chunks[i]['text'] for i in indices[0] if i != -1]
                if context_chunks:
                    return context_chunks
            except Exception as e:
                print(f"Embedding search failed: {e}")
        
        # Fallback to keyword search
        return self.keyword_search_medical(query, n_results)

    def keyword_search_medical(self, query: str, n_results: int) -> List[str]:
        """Medical-focused keyword search"""
        if not self.knowledge_chunks:
            return []
            
        query_words = set(query.lower().split())
        chunk_scores = []
        
        for chunk_info in self.knowledge_chunks:
            chunk_text = chunk_info['text']
            chunk_words = set(chunk_text.lower().split())
            
            word_overlap = len(query_words.intersection(chunk_words))
            base_score = word_overlap / len(query_words) if query_words else 0
            
            # Boost medical content
            medical_boost = 0
            if chunk_info.get('medical_focus') in ['pediatric_symptoms', 'treatments', 'diagnosis']:
                medical_boost = 0.5
                
            final_score = base_score + medical_boost
            
            if final_score > 0:
                chunk_scores.append((final_score, chunk_text))
        
        chunk_scores.sort(reverse=True)
        return [chunk for _, chunk in chunk_scores[:n_results]]

    def generate_biogpt_response(self, context: str, query: str) -> str:
        """Generate medical response using BioGPT"""
        if not self.model or not self.tokenizer:
            return "Medical model not available. Please check the setup."
            
        try:
            prompt = f"""Medical Context: {context[:800]}

Question: {query}

Medical Answer:"""

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            )
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if "Medical Answer:" in full_response:
                generated_response = full_response.split("Medical Answer:")[-1].strip()
            else:
                generated_response = full_response[len(prompt):].strip()
            
            return self.clean_medical_response(generated_response)
            
        except Exception as e:
            print(f"BioGPT generation failed: {e}")
            return self.fallback_response(context, query)

    def clean_medical_response(self, response: str) -> str:
        """Clean and format medical response"""
        sentences = re.split(r'[.!?]+', response)
        clean_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and not sentence.endswith(('and', 'or', 'but', 'however')):
                clean_sentences.append(sentence)
            if len(clean_sentences) >= 3:
                break
        
        if clean_sentences:
            cleaned = '. '.join(clean_sentences) + '.'
        else:
            cleaned = response[:200] + '...' if len(response) > 200 else response
            
        return cleaned

    def fallback_response(self, context: str, query: str) -> str:
        """Fallback response when BioGPT fails"""
        sentences = [s.strip() for s in context.split('.') if len(s.strip()) > 20]
        
        if sentences:
            response = sentences[0] + '.'
            if len(sentences) > 1:
                response += ' ' + sentences[1] + '.'
        else:
            response = context[:300] + '...'
            
        return response

    def handle_conversational_interactions(self, query: str) -> Optional[str]:
        """Handle conversational interactions"""
        query_lower = query.lower().strip()
        
        # Greetings
        if any(greeting in query_lower for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            return "👋 Hello! I'm BioGPT, your medical AI assistant specialized in pediatric medicine. Please upload your medical data file first, then ask me any health-related questions!"
        
        # Thanks
        if any(thanks in query_lower for thanks in ['thank you', 'thanks', 'thx', 'appreciate']):
            return "🙏 You're welcome! I'm glad I could help. Remember to always consult healthcare professionals for medical decisions. Feel free to ask more questions!"
        
        # Goodbyes
        if any(bye in query_lower for bye in ['bye', 'goodbye', 'see you', 'farewell']):
            return "👋 Goodbye! Take care of yourself and your family. Stay healthy! 🏥"
        
        # Help/About
        if any(help_word in query_lower for help_word in ['help', 'what can you do', 'how do you work']):
            return """🤖 **BioGPT Medical Assistant**

I'm an AI medical assistant that can help with:
• Pediatric medicine and children's health
• Medical symptoms and conditions
• Treatment information
• When to seek medical care

**How to use:**
1. Upload your medical data file using the file upload above
2. Ask specific medical questions
3. Get evidence-based medical information

⚠️ **Important:** I provide educational information only. Always consult healthcare professionals for medical advice."""

        return None

    def chat_interface(self, message: str, history: List[List[str]]) -> Tuple[str, List[List[str]]]:
        """Main chat interface for Gradio"""
        if not message.strip():
            return "", history
        
        # Check if data is loaded
        if not self.is_data_loaded:
            response = "⚠️ Please upload your medical data file first using the file upload above before asking questions."
            history.append([message, response])
            return "", history
        
        # Handle conversational interactions
        conversational_response = self.handle_conversational_interactions(message)
        if conversational_response:
            history.append([message, conversational_response])
            return "", history
        
        # Process medical query
        context = self.retrieve_medical_context(message)
        
        if not context:
            response = "I don't have specific information about this topic in my medical database. Please consult with a healthcare professional for personalized medical advice."
        else:
            main_context = '\n\n'.join(context)
            medical_response = self.generate_biogpt_response(main_context, message)
            response = f"🩺 **Medical Information:** {medical_response}\n\n⚠️ **Important:** This information is for educational purposes only. Always consult with qualified healthcare professionals for medical diagnosis, treatment, and personalized advice."
        
        # Add to conversation history
        self.conversation_history.append({
            'query': message,
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
        
        history.append([message, response])
        return "", history

# Initialize the chatbot
print("🚀 Initializing BioGPT Medical Chatbot...")
chatbot = GradioBioGPTChatbot(use_gpu=True, use_8bit=True)

def upload_and_process_file(file):
    """Handle file upload and processing"""
    if file is None:
        return "❌ No file uploaded."
    
    message, success = chatbot.load_medical_data_from_file(file.name)
    return message

# Create Gradio Interface
def create_gradio_interface():
    """Create and launch Gradio interface"""
    
    with gr.Blocks(
        title="🏥 BioGPT Medical Assistant",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .chat-message {
            border-radius: 10px !important;
        }
        """
    ) as demo:
        
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🏥 BioGPT Medical Assistant</h1>
            <p style="font-size: 18px; color: #666;">
                Professional AI Medical Chatbot powered by BioGPT-Large
            </p>
            <p style="color: #888;">
                ⚠️ For educational purposes only. Always consult healthcare professionals for medical advice.
            </p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<h3>📁 Upload Medical Data</h3>")
                file_upload = gr.File(
                    label="Upload Medical Text File (.txt)",
                    file_types=[".txt"],
                    type="file"
                )
                upload_status = gr.Textbox(
                    label="Upload Status",
                    value="📋 Please upload your medical data file to begin...",
                    interactive=False,
                    lines=3
                )
                
                gr.HTML("""
                <div style="margin-top: 20px; padding: 15px; background-color: #f0f8ff; border-radius: 10px;">
                    <h4>💡 How to Use:</h4>
                    <ol>
                        <li>Upload your medical text file (.txt format)</li>
                        <li>Wait for processing confirmation</li>
                        <li>Start asking medical questions!</li>
                    </ol>
                    
                    <h4>📝 Example Questions:</h4>
                    <ul>
                        <li>"What causes fever in children?"</li>
                        <li>"How to treat a persistent cough?"</li>
                        <li>"When should I call the doctor?"</li>
                        <li>"Signs of dehydration in infants?"</li>
                    </ul>
                </div>
                """)
            
            with gr.Column(scale=2):
                gr.HTML("<h3>💬 Medical Consultation</h3>")
                chatbot_interface = gr.Chatbot(
                    label="BioGPT Medical Chat",
                    height=500,
                    bubble_full_width=False
                )
                
                msg_input = gr.Textbox(
                    label="Your Medical Question",
                    placeholder="Ask me about pediatric health, symptoms, treatments, or when to seek care...",
                    lines=2
                )
                
                with gr.Row():
                    send_btn = gr.Button("🩺 Send Question", variant="primary")
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
        
        # Event handlers
        file_upload.change(
            fn=upload_and_process_file,
            inputs=[file_upload],
            outputs=[upload_status]
        )
        
        msg_input.submit(
            fn=chatbot.chat_interface,
            inputs=[msg_input, chatbot_interface],
            outputs=[msg_input, chatbot_interface]
        )
        
        send_btn.click(
            fn=chatbot.chat_interface,
            inputs=[msg_input, chatbot_interface],
            outputs=[msg_input, chatbot_interface]
        )
        
        clear_btn.click(
            fn=lambda: ([], ""),
            outputs=[chatbot_interface, msg_input]
        )
        
        gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; background-color: #fff3cd; border-radius: 10px;">
            <h4>⚠️ Medical Disclaimer</h4>
            <p>This AI assistant provides educational medical information only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with questions about medical conditions.</p>
        </div>
        """)
    
    return demo

if __name__ == "__main__":
    # Create and launch the Gradio interface
    demo = create_gradio_interface()
    
    print("🌐 Launching Gradio interface...")
    print("📋 Upload your medical data file and start chatting!")
    
    # Launch with public sharing (set share=False for local only)
    demo.launch(
        share=True,  # Set to False for local deployment only
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,  # Default Gradio port
        show_error=True,
        debug=True
    )