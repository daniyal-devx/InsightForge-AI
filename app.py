"""
InsightForge AI - Business Intelligence Chatbot
A production-ready RAG system for PDF-based business intelligence
"""

import gradio as gr
import os
import json
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import numpy as np
from pathlib import Path

# PDF Processing
import pdfplumber

# Embeddings and Vector Store
from sentence_transformers import SentenceTransformer
import faiss

# LLM Integration
from groq import Groq

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Application configuration"""
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    GROQ_MODEL = "llama3-8b-8192"
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 200
    TOP_K_RESULTS = 4
    MAX_HISTORY = 10
    EMBEDDING_DIMENSION = 384  # for all-MiniLM-L6-v2

# =============================================================================
# DOCUMENT PROCESSOR
# =============================================================================

class DocumentProcessor:
    """Handles PDF text extraction and chunking"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, any]]:
        """
        Extract text from PDF with page numbers
        
        Returns:
            List of dicts with 'page', 'text', 'filename'
        """
        pages_data = []
        filename = Path(pdf_path).name
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if text and text.strip():
                        pages_data.append({
                            'page': page_num,
                            'text': text.strip(),
                            'filename': filename
                        })
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            
        return pages_data
    
    @staticmethod
    def create_chunks(pages_data: List[Dict], 
                     chunk_size: int = Config.CHUNK_SIZE,
                     overlap: int = Config.CHUNK_OVERLAP) -> List[Dict]:
        """
        Create overlapping text chunks with metadata
        
        Returns:
            List of dicts with 'text', 'page', 'filename', 'chunk_id'
        """
        chunks = []
        chunk_id = 0
        
        for page_data in pages_data:
            text = page_data['text']
            page_num = page_data['page']
            filename = page_data['filename']
            
            # Split into sentences (simple approach)
            sentences = text.replace('\n', ' ').split('. ')
            
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                sentence_length = len(sentence)
                
                if current_length + sentence_length > chunk_size and current_chunk:
                    # Save current chunk
                    chunks.append({
                        'text': '. '.join(current_chunk) + '.',
                        'page': page_num,
                        'filename': filename,
                        'chunk_id': chunk_id
                    })
                    chunk_id += 1
                    
                    # Keep last few sentences for overlap
                    overlap_sentences = []
                    overlap_length = 0
                    for s in reversed(current_chunk):
                        if overlap_length + len(s) <= overlap:
                            overlap_sentences.insert(0, s)
                            overlap_length += len(s)
                        else:
                            break
                    
                    current_chunk = overlap_sentences
                    current_length = overlap_length
                
                current_chunk.append(sentence)
                current_length += sentence_length
            
            # Add remaining chunk
            if current_chunk:
                chunks.append({
                    'text': '. '.join(current_chunk) + '.',
                    'page': page_num,
                    'filename': filename,
                    'chunk_id': chunk_id
                })
                chunk_id += 1
        
        return chunks

# =============================================================================
# VECTOR STORE
# =============================================================================

class VectorStore:
    """FAISS-based vector store for semantic search"""
    
    def __init__(self, embedding_model_name: str = Config.EMBEDDING_MODEL):
        """Initialize embedding model and FAISS index"""
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.dimension = Config.EMBEDDING_DIMENSION
        self.index = None
        self.chunks = []
        self.is_ready = False
        
    def build_index(self, chunks: List[Dict]):
        """Build FAISS index from document chunks"""
        if not chunks:
            raise ValueError("No chunks provided to build index")
        
        self.chunks = chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings.astype('float32'))
        self.is_ready = True
        
        print(f"✓ Index built with {len(chunks)} chunks")
    
    def search(self, query: str, top_k: int = Config.TOP_K_RESULTS) -> List[Dict]:
        """
        Search for most relevant chunks
        
        Returns:
            List of chunks with similarity scores
        """
        if not self.is_ready:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        )
        
        # Search
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            top_k
        )
        
        # Prepare results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            chunk = self.chunks[idx].copy()
            chunk['similarity_score'] = float(1 / (1 + distance))  # Convert distance to similarity
            results.append(chunk)
        
        return results
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        if not self.is_ready:
            return {'status': 'Not initialized', 'chunks': 0, 'documents': 0}
        
        unique_docs = len(set(chunk['filename'] for chunk in self.chunks))
        
        return {
            'status': 'Ready',
            'chunks': len(self.chunks),
            'documents': unique_docs,
            'dimension': self.dimension
        }

# =============================================================================
# RAG CHATBOT
# =============================================================================

class RAGChatbot:
    """Main RAG chatbot with conversational memory"""
    
    def __init__(self, groq_api_key: str):
        """Initialize chatbot with Groq API"""
        self.groq_client = Groq(api_key=groq_api_key)
        self.vector_store = VectorStore()
        self.conversation_history = []
        self.query_log = []
        
    def process_pdfs(self, pdf_files: List[str]) -> str:
        """Process uploaded PDFs and build vector index"""
        if not pdf_files:
            return "⚠️ No PDF files uploaded"
        
        try:
            # Extract text from all PDFs
            all_pages_data = []
            for pdf_path in pdf_files:
                pages_data = DocumentProcessor.extract_text_from_pdf(pdf_path)
                all_pages_data.extend(pages_data)
            
            if not all_pages_data:
                return "⚠️ No text could be extracted from the PDFs"
            
            # Create chunks
            chunks = DocumentProcessor.create_chunks(all_pages_data)
            
            # Build vector index
            self.vector_store.build_index(chunks)
            
            # Clear conversation history for new documents
            self.conversation_history = []
            
            stats = self.vector_store.get_stats()
            
            return f"""✅ **Documents processed successfully!**
            
📊 **Statistics:**
- Documents loaded: {stats['documents']}
- Total chunks: {stats['chunks']}
- Embedding dimension: {stats['dimension']}

💬 Ready to answer your business intelligence questions!"""
            
        except Exception as e:
            return f"❌ Error processing PDFs: {str(e)}"
    
    def generate_response(self, query: str, history: List[List[str]]) -> Tuple[str, str]:
        """
        Generate response using RAG
        
        Returns:
            (answer, sources)
        """
        if not self.vector_store.is_ready:
            return "⚠️ Please upload and process PDF documents first.", ""
        
        try:
            # Log query
            self.query_log.append({
                'timestamp': datetime.now().isoformat(),
                'query': query
            })
            
            # Retrieve relevant chunks
            relevant_chunks = self.vector_store.search(query, top_k=Config.TOP_K_RESULTS)
            
            if not relevant_chunks:
                return "⚠️ No relevant information found in the documents.", ""
            
            # Build context from chunks
            context_parts = []
            for i, chunk in enumerate(relevant_chunks, 1):
                context_parts.append(
                    f"[Source {i}] {chunk['filename']} (Page {chunk['page']}):\n{chunk['text']}"
                )
            
            context = "\n\n".join(context_parts)
            
            # Build conversation context
            conversation_context = ""
            if self.conversation_history:
                recent_history = self.conversation_history[-Config.MAX_HISTORY:]
                conversation_context = "\n".join([
                    f"User: {h['query']}\nAssistant: {h['response']}"
                    for h in recent_history
                ])
            
            # Create prompt
            system_prompt = """You are InsightForge AI, an expert business intelligence analyst. 
Your role is to provide clear, actionable insights based on the provided business documents.

Guidelines:
- Provide specific, data-driven insights
- Cite sources when making claims
- Highlight key trends and patterns
- Offer strategic recommendations when appropriate
- Be concise but comprehensive
- If the information isn't in the context, say so clearly"""

            user_prompt = f"""Based on the following business documents, answer this question:

Question: {query}

Relevant Context:
{context}

{"Previous Conversation:" + conversation_context if conversation_context else ""}

Provide a clear, professional answer with specific insights from the documents."""

            # Call Groq API
            response = self.groq_client.chat.completions.create(
                model=Config.GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1024,
                top_p=0.9
            )
            
            answer = response.choices[0].message.content
            
            # Add disclaimer
            answer += "\n\n---\n⚠️ *This analysis is based on the uploaded documents. Always verify critical business decisions with additional sources and expert consultation.*"
            
            # Store in conversation history
            self.conversation_history.append({
                'query': query,
                'response': answer,
                'timestamp': datetime.now().isoformat()
            })
            
            # Build sources section
            sources = "**📚 Sources:**\n\n"
            for i, chunk in enumerate(relevant_chunks, 1):
                sources += f"**[{i}]** {chunk['filename']} - Page {chunk['page']} (Relevance: {chunk['similarity_score']:.2%})\n"
                sources += f"```\n{chunk['text'][:200]}...\n```\n\n"
            
            return answer, sources
            
        except Exception as e:
            return f"❌ Error generating response: {str(e)}", ""
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        return "✅ Conversation history cleared"
    
    def get_chat_history_export(self) -> str:
        """Export chat history as JSON"""
        if not self.conversation_history:
            return "No chat history available"
        
        export_data = {
            'export_date': datetime.now().isoformat(),
            'total_queries': len(self.conversation_history),
            'conversation': self.conversation_history
        }
        
        return json.dumps(export_data, indent=2)
    
    def get_document_summary(self) -> str:
        """Get summary of loaded documents"""
        stats = self.vector_store.get_stats()
        
        if stats['status'] != 'Ready':
            return "No documents loaded"
        
        # Get unique documents
        doc_info = {}
        for chunk in self.vector_store.chunks:
            filename = chunk['filename']
            if filename not in doc_info:
                doc_info[filename] = {'pages': set(), 'chunks': 0}
            doc_info[filename]['pages'].add(chunk['page'])
            doc_info[filename]['chunks'] += 1
        
        summary = "📄 **Loaded Documents:**\n\n"
        for filename, info in doc_info.items():
            summary += f"**{filename}**\n"
            summary += f"- Pages: {len(info['pages'])}\n"
            summary += f"- Chunks: {info['chunks']}\n\n"
        
        return summary

# =============================================================================
# GRADIO INTERFACE
# =============================================================================

def create_interface():
    """Create Gradio interface"""
    
    # Get API key from environment
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    
    # Initialize chatbot
    chatbot = RAGChatbot(groq_api_key)
    
    # Custom CSS for professional dark theme
    custom_css = """
    .gradio-container {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
    }
    .main-header {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .upload-section {
        background: rgba(255,255,255,0.05);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .sources-section {
        background: rgba(102, 126, 234, 0.1);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    """
    
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="purple")) as demo:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1 style="color: white; margin: 0; font-size: 2.5rem;">🔮 InsightForge AI</h1>
            <p style="color: rgba(255,255,255,0.9); margin-top: 0.5rem; font-size: 1.1rem;">
                Business Intelligence Chatbot powered by RAG
            </p>
        </div>
        """)
        
        with gr.Row():
            # Left Sidebar
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload Documents")
                
                pdf_upload = gr.File(
                    label="Upload PDF Files",
                    file_count="multiple",
                    file_types=[".pdf"],
                    type="filepath"
                )
                
                upload_btn = gr.Button("🚀 Process Documents", variant="primary", size="lg")
                upload_status = gr.Markdown("Awaiting documents...")
                
                gr.Markdown("---")
                
                doc_summary = gr.Markdown("### 📊 Document Summary\nNo documents loaded")
                
                gr.Markdown("---")
                
                clear_btn = gr.Button("🗑️ Clear Chat History", variant="secondary")
                clear_status = gr.Markdown("")
                
                gr.Markdown("---")
                
                download_btn = gr.Button("💾 Download Chat History", variant="secondary")
                history_output = gr.Textbox(label="Chat History (JSON)", visible=False)
            
            # Main Chat Area
            with gr.Column(scale=3):
                chatbot_interface = gr.Chatbot(
                    label="💬 Conversation",
                    height=500,
                    show_copy_button=True,
                    avatar_images=(None, "🤖")
                )
                
                with gr.Row():
                    query_input = gr.Textbox(
                        label="Ask a question about your business documents",
                        placeholder="e.g., What are the key revenue trends in Q3? What strategic recommendations are mentioned?",
                        lines=2,
                        scale=4
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)
                
                sources_display = gr.Markdown("### 📚 Sources\nSources will appear here after asking questions")
        
        # Event Handlers
        def process_pdfs_handler(files):
            if not files:
                return "⚠️ Please upload PDF files first", "No documents loaded"
            
            status = chatbot.process_pdfs(files)
            summary = chatbot.get_document_summary()
            return status, summary
        
        def respond(message, chat_history):
            if not message.strip():
                return chat_history, ""
            
            answer, sources = chatbot.generate_response(message, chat_history)
            chat_history.append([message, answer])
            return chat_history, sources
        
        def clear_history_handler():
            chatbot.clear_history()
            return [], "✅ Conversation cleared", "### 📚 Sources\nSources will appear here after asking questions"
        
        def export_history():
            return chatbot.get_chat_history_export()
        
        # Connect events
        upload_btn.click(
            fn=process_pdfs_handler,
            inputs=[pdf_upload],
            outputs=[upload_status, doc_summary]
        )
        
        submit_btn.click(
            fn=respond,
            inputs=[query_input, chatbot_interface],
            outputs=[chatbot_interface, sources_display]
        ).then(
            lambda: "",
            outputs=[query_input]
        )
        
        query_input.submit(
            fn=respond,
            inputs=[query_input, chatbot_interface],
            outputs=[chatbot_interface, sources_display]
        ).then(
            lambda: "",
            outputs=[query_input]
        )
        
        clear_btn.click(
            fn=clear_history_handler,
            outputs=[chatbot_interface, clear_status, sources_display]
        )
        
        download_btn.click(
            fn=export_history,
            outputs=[history_output]
        ).then(
            lambda: gr.update(visible=True),
            outputs=[history_output]
        )
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 2rem; color: rgba(255,255,255,0.6);">
            <p>Built with Groq AI • Powered by FAISS & Sentence Transformers</p>
            <p style="font-size: 0.9rem;">⚡ Production-Ready RAG System for Business Intelligence</p>
        </div>
        """)
    
    return demo

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
