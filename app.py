from flask import Flask, request, jsonify, send_from_directory, url_for
from flask_cors import CORS
import fitz
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import tempfile
from dotenv import load_dotenv
import json
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from collections import defaultdict
import re
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Load spaCy model with error handling
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy English language model...")
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Global variables for storing conversation state
conversation = None
vector_store = None
chat_history = []

def extract_metadata(text, page_num):
    """Extract metadata from text including entities, key concepts, and relationships."""
    doc = nlp(text)
    
    # Extract named entities
    entities = [ent.text for ent in doc.ents]
    
    # Extract key concepts (nouns and noun phrases)
    concepts = [chunk.text for chunk in doc.noun_chunks]
    
    # Extract relationships (subject-verb-object patterns)
    relationships = []
    for sent in doc.sents:
        for token in sent:
            if token.dep_ in ('nsubj', 'nsubjpass'):
                subject = token.text
                verb = token.head.text
                for child in token.head.children:
                    if child.dep_ in ('dobj', 'pobj'):
                        relationships.append(f"{subject} {verb} {child.text}")
    
    # Extract dates and numbers
    dates = re.findall(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}', text)
    numbers = re.findall(r'\b\d+\b', text)
    
    return {
        'page': page_num,
        'entities': entities,
        'concepts': concepts,
        'relationships': relationships,
        'dates': dates,
        'numbers': numbers
    }

def process_pdf(pdf_file):
    """Process PDF file and return chunks with metadata."""
    tmp_path = None
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(pdf_file.read())

        # Extract text from PDF
        doc = fitz.open(tmp_path)
        chunks_with_metadata = []
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            
            # Split text into sentences for better context
            sentences = sent_tokenize(text)
            
            # Create chunks with overlapping context
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                current_chunk.append(sentence)
                current_length += len(sentence)
                
                if current_length >= 1000:  # Chunk size threshold
                    chunk_text = ' '.join(current_chunk)
                    metadata = extract_metadata(chunk_text, page_num + 1)
                    
                    chunks_with_metadata.append({
                        'text': chunk_text,
                        'metadata': metadata
                    })
                    
                    # Keep last few sentences for context
                    current_chunk = current_chunk[-3:]  # Keep last 3 sentences
                    current_length = sum(len(s) for s in current_chunk)
        
        # Close the PDF document
        doc.close()
        
        return chunks_with_metadata
    
    finally:
        # Clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Could not delete temporary file {tmp_path}: {str(e)}")

def create_vector_store(chunks_with_metadata):
    """Create vector store from text chunks with metadata."""
    try:
        # Initialize embeddings with proper configuration
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",  # Using the latest embedding model
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            chunk_size=1000  # Add chunk size for better performance
        )
        
        # Create documents with metadata
        documents = []
        for chunk in chunks_with_metadata:
            # Combine text with metadata for better semantic search
            enhanced_text = f"{chunk['text']} {' '.join(chunk['metadata']['concepts'])} {' '.join(chunk['metadata']['entities'])}"
            # Create a proper Document object
            doc = Document(
                page_content=enhanced_text,
                metadata=chunk['metadata']
            )
            documents.append(doc)
        
        # Create FAISS vector store
        vector_store = FAISS.from_documents(
            documents,
            embeddings
        )
        
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}", exc_info=True)
        raise

def setup_conversation_chain(vector_store):
    """Set up the conversation chain with memory and enhanced retrieval."""
    llm = ChatOpenAI(temperature=0.7)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # Explicitly set which output to store in memory
    )
    
    # Create a custom retriever that considers metadata
    retriever = vector_store.as_retriever(
        search_kwargs={
            'k': 5,  # Number of chunks to retrieve
            'fetch_k': 20,  # Number of chunks to fetch before filtering
            'score_threshold': 0.5  # Minimum similarity score
        }
    )
    
    # Create the prompt template with proper context formatting
    prompt_template = PromptTemplate(
        template="""You are an AI assistant analyzing a document. Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        Use the metadata and relationships to provide a comprehensive answer that connects different concepts.
        
        Context: {context}
        
        Chat History:
        {chat_history}
        
        Question: {question}
        
        Answer:""",
        input_variables=["context", "chat_history", "question"]
    )
    
    # Create conversation chain with custom prompt and proper document handling
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,  # Return source documents for better context
        combine_docs_chain_kwargs={
            'prompt': prompt_template,
            'document_variable_name': 'context'  # Explicitly set the document variable name
        },
        verbose=True  # Enable verbose logging for debugging
    )
    
    return conversation_chain

@app.route('/')
def serve_index():
    logger.debug("Serving index.html")
    return send_from_directory('static', 'index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global conversation, vector_store, chat_history
    
    logger.debug("Received file upload request")
    
    if 'file' not in request.files:
        logger.error("No file in request")
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.error("Empty filename")
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        logger.debug(f"Processing file: {file.filename}")
        # Process PDF and create vector store
        chunks = process_pdf(file)
        logger.debug(f"Created {len(chunks)} chunks")
        
        vector_store = create_vector_store(chunks)
        logger.debug("Created vector store")
        
        conversation = setup_conversation_chain(vector_store)
        logger.debug("Set up conversation chain")
        
        chat_history = []
        
        return jsonify({'message': 'PDF processed successfully'}), 200
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    global conversation, chat_history
    
    logger.debug("Received question request")
    
    if not conversation:
        logger.error("No conversation initialized")
        return jsonify({'error': 'Please upload a PDF first'}), 400
    
    data = request.get_json()
    if not data or 'question' not in data:
        logger.error("No question in request")
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        logger.debug(f"Processing question: {data['question']}")
        # Get response from conversation chain
        response = conversation({"question": data['question']})
        
        # Store chat history
        chat_history.append({
            'question': data['question'],
            'answer': response['answer'],
            'sources': [
                {
                    'page': doc.metadata['page'],
                    'content': doc.page_content[:200] + '...'
                } for doc in response.get('source_documents', [])
            ]
        })
        
        logger.debug("Generated response successfully")
        return jsonify({
            'answer': response['answer'],
            'chat_history': chat_history
        }), 200
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
