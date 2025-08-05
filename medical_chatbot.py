import os
import re
import torch
import warnings
import numpy as np
import faiss
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import time
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class ColabBioGPTChatbot:
    def __init__(self, use_gpu=True, use_8bit=True):
        """Initialize BioGPT chatbot optimized for Hugging Face Spaces"""
        print("🏥 Initializing Medical Chatbot...")
        self.use_gpu = use_gpu
        self.use_8bit = use_8bit
        self.device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        print(f"🖥️ Using device: {self.device}")
        
        self.tokenizer = None
        self.model = None
        self.knowledge_chunks = []
        self.conversation_history = []
        self.embedding_model = None
        self.faiss_index = None
        self.faiss_ready = False
        self.use_embeddings = True
        
        # Initialize components
        self.setup_biogpt()
        self.load_sentence_transformer()
        
    def setup_biogpt(self):
        """Setup BioGPT model with fallback to base BioGPT if Large fails"""
        print("🧠 Loading BioGPT model...")
        
        try:
            # Try BioGPT-Large first
            model_name = "microsoft/BioGPT-Large"
            print(f"Attempting to load {model_name}...")
            
            if self.use_8bit and self.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                )
            else:
                quantization_config = None
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cuda" and quantization_config is None:
                self.model = self.model.to(self.device)
                
            print("✅ BioGPT-Large loaded successfully!")
            
        except Exception as e:
            print(f"❌ BioGPT-Large loading failed: {e}")
            print("🔁 Falling back to base BioGPT...")
            self.setup_fallback_biogpt()
    
    def setup_fallback_biogpt(self):
        """Fallback to microsoft/BioGPT if BioGPT-Large fails"""
        try:
            model_name = "microsoft/BioGPT"
            print(f"Loading fallback model: {model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cuda":
                self.model = self.model.to(self.device)
                
            print("✅ Base BioGPT model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Failed to load fallback BioGPT: {e}")
            self.model = None
            self.tokenizer = None
    
    def load_sentence_transformer(self):
        """Load sentence transformer for embeddings"""
        try:
            print("🔮 Loading sentence transformer...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize FAISS index (will be populated when data is loaded)
            embedding_dim = 384  # Dimension for all-MiniLM-L6-v2
            self.faiss_index = faiss.IndexFlatL2(embedding_dim)
            self.faiss_ready = True
            print("✅ Sentence transformer and FAISS index ready!")
            
        except Exception as e:
            print(f"❌ Failed to load sentence transformer: {e}")
            self.use_embeddings = False
            self.faiss_ready = False

    def load_medical_data(self, file_path):
        """Load and process medical data"""
        print(f"📖 Loading medical data from {file_path}...")
        
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File {file_path} not found")
                
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            print(f"📄 File loaded: {len(text):,} characters")
            
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            raise ValueError(f"Failed to load medical data: {e}")
        
        # Create chunks
        print("📝 Creating medical chunks...")
        chunks = self.create_medical_chunks(text)
        print(f"📋 Created {len(chunks)} medical chunks")
        
        self.knowledge_chunks = chunks
        
        # Generate embeddings if available
        if self.use_embeddings and self.embedding_model and self.faiss_ready:
            try:
                self.generate_embeddings_with_progress(chunks)
                print("✅ Medical data loaded with embeddings!")
            except Exception as e:
                print(f"⚠️ Embedding generation failed: {e}")
                print("✅ Medical data loaded (keyword search mode)")
        else:
            print("✅ Medical data loaded (keyword search mode)")
    
    def create_medical_chunks(self, text: str, chunk_size: int = 400) -> List[Dict]:
        """Create medically-optimized text chunks"""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
        
        chunk_id = 0
        for paragraph in paragraphs:
            if len(paragraph.split()) <= chunk_size:
                chunks.append({
                    'id': chunk_id,
                    'text': paragraph,
                    'medical_focus': self.identify_medical_focus(paragraph)
                })
                chunk_id += 1
            else:
                # Split large paragraphs by sentences
                sentences = re.split(r'[.!?]+', paragraph)
                current_chunk = ""
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    if len(current_chunk.split()) + len(sentence.split()) <= chunk_size:
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
        
        return chunks
    
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
    
    def generate_embeddings_with_progress(self, chunks: List[Dict]):
        """Generate embeddings and add to FAISS index"""
        print("🔮 Generating embeddings...")
        
        try:
            texts = [chunk['text'] for chunk in chunks]
            
            # Generate embeddings in batches
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_embeddings = self.embedding_model.encode(batch_texts, show_progress_bar=False)
                all_embeddings.extend(batch_embeddings)
                
                progress = min(i + batch_size, len(texts))
                print(f"   Progress: {progress}/{len(texts)} chunks processed", end='\r')
            
            print(f"\n   ✅ Generated embeddings for {len(texts)} chunks")
            
            # Add to FAISS index
            embeddings_array = np.array(all_embeddings).astype('float32')
            self.faiss_index.add(embeddings_array)
            print("✅ Embeddings added to FAISS index!")
            
        except Exception as e:
            print(f"❌ Embedding generation failed: {e}")
            raise
    
    def retrieve_medical_context(self, query: str, n_results: int = 3) -> List[str]:
        """Retrieve relevant medical context"""
        if self.use_embeddings and self.embedding_model and self.faiss_ready and self.faiss_index.ntotal > 0:
            try:
                # Generate query embedding
                query_embedding = self.embedding_model.encode([query])
                
                # Search FAISS index
                distances, indices = self.faiss_index.search(
                    np.array(query_embedding).astype('float32'), 
                    min(n_results, self.faiss_index.ntotal)
                )
                
                # Get relevant chunks
                context_chunks = []
                for idx in indices[0]:
                    if idx != -1 and idx < len(self.knowledge_chunks):
                        context_chunks.append(self.knowledge_chunks[idx]['text'])
                
                if context_chunks:
                    return context_chunks
                    
            except Exception as e:
                print(f"⚠️ Embedding search failed: {e}")
        
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
            
            # Calculate relevance score
            word_overlap = len(query_words.intersection(chunk_words))
            base_score = word_overlap / len(query_words) if query_words else 0
            
            # Boost medical content
            medical_boost = 0
            if chunk_info.get('medical_focus') in ['pediatric_symptoms', 'treatments', 'diagnosis']:
                medical_boost = 0.3
            
            final_score = base_score + medical_boost
            
            if final_score > 0:
                chunk_scores.append((final_score, chunk_text))
        
        # Return top matches
        chunk_scores.sort(reverse=True)
        return [chunk for _, chunk in chunk_scores[:n_results]]
    
    def generate_biogpt_response(self, context: str, query: str) -> str:
        """Generate medical response using context directly (BioGPT bypass)"""
        # BioGPT is giving poor responses, so use the retrieved context directly
        return self.create_context_based_response(context, query)
    
    def create_context_based_response(self, context: str, query: str) -> str:
        """Create response directly from medical context"""
        if not context:
            return "I don't have specific information about this topic in my medical database."
        
        # Split context into sentences
        sentences = [s.strip() + '.' for s in context.split('.') if len(s.strip()) > 15]
        
        # Find sentences most relevant to the query
        query_words = set(query.lower().split())
        scored_sentences = []
        
        for sentence in sentences[:20]:  # Increased from 15 to 20
            sentence_words = set(sentence.lower().split())
            # Score based on word overlap
            score = len(query_words.intersection(sentence_words))
            if score > 0:
                scored_sentences.append((score, sentence))
        
        # Sort by relevance and take top sentences
        scored_sentences.sort(reverse=True)
        
        if scored_sentences:
            # Take top 3-4 most relevant sentences for better coverage
            response_sentences = [sent for _, sent in scored_sentences[:4]]
            response = ' '.join(response_sentences)
        else:
            # Fallback to first few sentences
            response = ' '.join(sentences[:3])
        
        # Clean up the response
        response = re.sub(r'\s+', ' ', response).strip()
        
        return response[:500] + '...' if len(response) > 500 else response  # Increased from 400
    
    def clean_medical_response(self, response: str) -> str:
        """Clean and format medical response"""
        # Remove training artifacts and unwanted symbols
        response = re.sub(r'<[^>]*>', '', response)  # Remove HTML-like tags
        response = re.sub(r'▃+', '', response)  # Remove block symbols
        response = re.sub(r'FREETEXT|INTRO|/FREETEXT|/INTRO', '', response)  # Remove training markers
        response = re.sub(r'\s+', ' ', response)  # Clean up whitespace
        response = response.strip()
        
        # Split into sentences and keep only complete, relevant ones
        sentences = re.split(r'[.!?]+', response)
        clean_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            # Skip very short sentences and those with artifacts
            if len(sentence) > 15 and not any(artifact in sentence.lower() for artifact in ['▃', '<', '>', 'freetext']):
                clean_sentences.append(sentence)
            if len(clean_sentences) >= 2:  # Limit to 2 good sentences
                break
        
        if clean_sentences:
            cleaned = '. '.join(clean_sentences) + '.'
        else:
            # Fallback to first 150 characters if no good sentences found
            cleaned = response[:150].strip()
            if cleaned and not cleaned.endswith('.'):
                cleaned += '.'
        
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
        
        # Only match very specific greeting patterns (must be standalone)
        if re.match(r'^\s*(hello|hi|hey)\s*$', query_lower):
            return "👋 Hello! I'm your pediatric medical AI assistant. How can I help you with medical questions today?"
        
        if re.match(r'^\s*(good morning|good afternoon|good evening)\s*$', query_lower):
            return "👋 Hello! I'm your pediatric medical AI assistant. How can I help you with medical questions today?"
        
        # Only match very specific thanks patterns (must be standalone)
        if re.match(r'^\s*(thank you|thanks|thx)\s*$', query_lower):
            return "🙏 You're welcome! I'm glad I could help. Remember to consult healthcare professionals for medical decisions. What else can I help you with?"
        
        # Only match very specific goodbye patterns (must be standalone)
        if re.match(r'^\s*(bye|goodbye)\s*$', query_lower):
            return "👋 Goodbye! Take care and remember to consult healthcare professionals for any medical concerns. Stay healthy!"
        
        return None
    
    def chat(self, query: str) -> str:
        """Main chat function"""
        if not query.strip():
            return "Hello! I'm your pediatric medical AI assistant. How can I help you today?"
        
        # Handle conversational interactions
        conversational_response = self.handle_conversational_interactions(query)
        if conversational_response:
            return conversational_response
        
        if not self.knowledge_chunks:
            return "Please load medical data first to access the medical knowledge base."
        
        if not self.model or not self.tokenizer:
            return "Medical model not available. Please check the setup and try again."
        
        # Retrieve context
        context = self.retrieve_medical_context(query)
        
        if not context:
            return "I don't have specific information about this topic in my medical database. Please consult with a healthcare professional for personalized medical advice."
        
        # Generate response
        main_context = '\n\n'.join(context)
        response = self.generate_biogpt_response(main_context, query)
        
        # Format final response
        final_response = f"🩺 **Medical Information:** {response}\n\n⚠️ **Important:** This information is for educational purposes only. Always consult with qualified healthcare professionals for medical diagnosis, treatment, and personalized advice."
        
        return final_response,
        r'^\s*(good morning|good afternoon|good evening)\s*$',
    
    def chat(self, query: str) -> str:
        """Main chat function"""
        if not query.strip():
            return "Hello! I'm your pediatric medical AI assistant. How can I help you today?"
        
        # Handle conversational interactions
        conversational_response = self.handle_conversational_interactions(query)
        if conversational_response:
            return conversational_response
        
        if not self.knowledge_chunks:
            return "Please load medical data first to access the medical knowledge base."
        
        if not self.model or not self.tokenizer:
            return "Medical model not available. Please check the setup and try again."
        
        # Retrieve context
        context = self.retrieve_medical_context(query)
        
        if not context:
            return "I don't have specific information about this topic in my medical database. Please consult with a healthcare professional for personalized medical advice."
        
        # Generate response
        main_context = '\n\n'.join(context)
        response = self.generate_biogpt_response(main_context, query)
        
        # Format final response
        final_response = f"🩺 **Medical Information:** {response}\n\n⚠️ **Important:** This information is for educational purposes only. Always consult with qualified healthcare professionals for medical diagnosis, treatment, and personalized advice."
        
        return final_response,
        r'^\s*(hi there|hello there)\s*$'
    
    def chat(self, query: str) -> str:
        """Main chat function"""
        if not query.strip():
            return "Hello! I'm your pediatric medical AI assistant. How can I help you today?"
        
        # Handle conversational interactions
        conversational_response = self.handle_conversational_interactions(query)
        if conversational_response:
            return conversational_response
        
        if not self.knowledge_chunks:
            return "Please load medical data first to access the medical knowledge base."
        
        if not self.model or not self.tokenizer:
            return "Medical model not available. Please check the setup and try again."
        
        # Retrieve context
        context = self.retrieve_medical_context(query)
        
        if not context:
            return "I don't have specific information about this topic in my medical database. Please consult with a healthcare professional for personalized medical advice."
        
        # Generate response
        main_context = '\n\n'.join(context)
        response = self.generate_biogpt_response(main_context, query)
        
        # Format final response
        final_response = f"🩺 **Medical Information:** {response}\n\n⚠️ **Important:** This information is for educational purposes only. Always consult with qualified healthcare professionals for medical diagnosis, treatment, and personalized advice."
        
        return final_response
        
        
        for pattern in greeting_patterns:
            if re.match(pattern, query_lower):
                return "👋 Hello! I'm your pediatric medical AI assistant. How can I help you with medical questions today?"
        
        # Only match very specific thanks patterns (must be standalone)
        thanks_patterns = [
            r'^\s*(thank you|thanks|thx)\s*$'
        ]
    
    def chat(self, query: str) -> str:
        """Main chat function"""
        if not query.strip():
            return "Hello! I'm your pediatric medical AI assistant. How can I help you today?"
        
        # Handle conversational interactions
        conversational_response = self.handle_conversational_interactions(query)
        if conversational_response:
            return conversational_response
        
        if not self.knowledge_chunks:
            return "Please load medical data first to access the medical knowledge base."
        
        if not self.model or not self.tokenizer:
            return "Medical model not available. Please check the setup and try again."
        
        # Retrieve context
        context = self.retrieve_medical_context(query)
        
        if not context:
            return "I don't have specific information about this topic in my medical database. Please consult with a healthcare professional for personalized medical advice."
        
        # Generate response
        main_context = '\n\n'.join(context)
        response = self.generate_biogpt_response(main_context, query)
        
        # Format final response
        final_response = f"🩺 **Medical Information:** {response}\n\n⚠️ **Important:** This information is for educational purposes only. Always consult with qualified healthcare professionals for medical diagnosis, treatment, and personalized advice."
        
        return final_response,
        r'^\s*(thank you so much|thanks a lot)\s*$'
    
    def chat(self, query: str) -> str:
        """Main chat function"""
        if not query.strip():
            return "Hello! I'm your pediatric medical AI assistant. How can I help you today?"
        
        # Handle conversational interactions
        conversational_response = self.handle_conversational_interactions(query)
        if conversational_response:
            return conversational_response
        
        if not self.knowledge_chunks:
            return "Please load medical data first to access the medical knowledge base."
        
        if not self.model or not self.tokenizer:
            return "Medical model not available. Please check the setup and try again."
        
        # Retrieve context
        context = self.retrieve_medical_context(query)
        
        if not context:
            return "I don't have specific information about this topic in my medical database. Please consult with a healthcare professional for personalized medical advice."
        
        # Generate response
        main_context = '\n\n'.join(context)
        response = self.generate_biogpt_response(main_context, query)
        
        # Format final response
        final_response = f"🩺 **Medical Information:** {response}\n\n⚠️ **Important:** This information is for educational purposes only. Always consult with qualified healthcare professionals for medical diagnosis, treatment, and personalized advice."
        
        return final_response
        
        
        for pattern in thanks_patterns:
            if re.match(pattern, query_lower):
                return "🙏 You're welcome! I'm glad I could help. Remember to consult healthcare professionals for medical decisions. What else can I help you with?"
        
        # Only match very specific goodbye patterns (must be standalone)
        goodbye_patterns = [
        r'^\s*(bye|goodbye)\s*$'
        ]

    def chat(self, query: str) -> str:
        """Main chat function"""
        if not query.strip():
            return "Hello! I'm your pediatric medical AI assistant. How can I help you today?"
        
        # Handle conversational interactions
        conversational_response = self.handle_conversational_interactions(query)
        if conversational_response:
            return conversational_response
        
        if not self.knowledge_chunks:
            return "Please load medical data first to access the medical knowledge base."
        
        if not self.model or not self.tokenizer:
            return "Medical model not available. Please check the setup and try again."
        
        # Retrieve context
        context = self.retrieve_medical_context(query)
        
        if not context:
            return "I don't have specific information about this topic in my medical database. Please consult with a healthcare professional for personalized medical advice."
        
        # Generate response
        main_context = '\n\n'.join(context)
        response = self.generate_biogpt_response(main_context, query)
        
        # Format final response
        final_response = f"🩺 **Medical Information:** {response}\n\n⚠️ **Important:** This information is for educational purposes only. Always consult with qualified healthcare professionals for medical diagnosis, treatment, and personalized advice."
        
        return final_response,
        r'^\s*(see you later|see ya)\s*$'
    
    def chat(self, query: str) -> str:
        """Main chat function"""
        if not query.strip():
            return "Hello! I'm your pediatric medical AI assistant. How can I help you today?"
        
        # Handle conversational interactions
        conversational_response = self.handle_conversational_interactions(query)
        if conversational_response:
            return conversational_response
        
        if not self.knowledge_chunks:
            return "Please load medical data first to access the medical knowledge base."
        
        if not self.model or not self.tokenizer:
            return "Medical model not available. Please check the setup and try again."
        
        # Retrieve context
        context = self.retrieve_medical_context(query)
        
        if not context:
            return "I don't have specific information about this topic in my medical database. Please consult with a healthcare professional for personalized medical advice."
        
        # Generate response
        main_context = '\n\n'.join(context)
        response = self.generate_biogpt_response(main_context, query)
        
        # Format final response
        final_response = f"🩺 **Medical Information:** {response}\n\n⚠️ **Important:** This information is for educational purposes only. Always consult with qualified healthcare professionals for medical diagnosis, treatment, and personalized advice."
        
        return final_response,
        r'^\s*(have a good day|take care)\s*$'
    
    def chat(self, query: str) -> str:
        """Main chat function"""
        if not query.strip():
            return "Hello! I'm your pediatric medical AI assistant. How can I help you today?"
        
        # Handle conversational interactions
        conversational_response = self.handle_conversational_interactions(query)
        if conversational_response:
            return conversational_response
        
        if not self.knowledge_chunks:
            return "Please load medical data first to access the medical knowledge base."
        
        if not self.model or not self.tokenizer:
            return "Medical model not available. Please check the setup and try again."
        
        # Retrieve context
        context = self.retrieve_medical_context(query)
        
        if not context:
            return "I don't have specific information about this topic in my medical database. Please consult with a healthcare professional for personalized medical advice."
        
        # Generate response
        main_context = '\n\n'.join(context)
        response = self.generate_biogpt_response(main_context, query)
        
        # Format final response
        final_response = f"🩺 **Medical Information:** {response}\n\n⚠️ **Important:** This information is for educational purposes only. Always consult with qualified healthcare professionals for medical diagnosis, treatment, and personalized advice."
        
        return final_response
        
        
        for pattern in goodbye_patterns:
            if re.match(pattern, query_lower):
                return "👋 Goodbye! Take care and remember to consult healthcare professionals for any medical concerns. Stay healthy!"
        
        return None
        
    
    def chat(self, query: str) -> str:
        """Main chat function"""
        if not query.strip():
            return "Hello! I'm your pediatric medical AI assistant. How can I help you today?"
        
        # Handle conversational interactions
        conversational_response = self.handle_conversational_interactions(query)
        if conversational_response:
            return conversational_response
        
        if not self.knowledge_chunks:
            return "Please load medical data first to access the medical knowledge base."
        
        if not self.model or not self.tokenizer:
            return "Medical model not available. Please check the setup and try again."
        
        # Retrieve context
        context = self.retrieve_medical_context(query)
        
        if not context:
            return "I don't have specific information about this topic in my medical database. Please consult with a healthcare professional for personalized medical advice."
        
        # Generate response
        main_context = '\n\n'.join(context)
        response = self.generate_biogpt_response(main_context, query)
        
        # Format final response
        final_response = f"🩺 **Medical Information:** {response}\n\n⚠️ **Important:** This information is for educational purposes only. Always consult with qualified healthcare professionals for medical diagnosis, treatment, and personalized advice."
        
        return final_response