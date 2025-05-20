from flask import Flask, request, jsonify, send_from_directory, url_for
from flask_cors import CORS
import fitz
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
# Remove ChatOpenAI and import ChatGoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI 
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
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

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
    
    entities = [ent.text for ent in doc.ents]
    concepts = [chunk.text for chunk in doc.noun_chunks]
    relationships = []
    for sent in doc.sents:
        for token in sent:
            if token.dep_ in ('nsubj', 'nsubjpass'):
                subject = token.text
                verb = token.head.text
                for child in token.head.children:
                    if child.dep_ in ('dobj', 'pobj'):
                        relationships.append(f"{subject} {verb} {child.text}")
    
    dates = re.findall(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}', text)
    numbers = re.findall(r'\d+', text)
    
    return {
        'page': page_num,
        'entities': entities,
        'concepts': concepts,
        'relationships': relationships,
        'dates': dates,
        'numbers': numbers
    }

def extract_table_data(page):
    """Extract tables from a PDF page using PyMuPDF's table detection."""
    tables = []
    try:
        # Get the page's tables
        tab = page.find_tables()
        if tab.tables:
            logger.debug(f"Found {len(tab.tables)} tables on page")
            for table in tab.tables:
                table_data = []
                for row in table.extract():
                    # Clean and process each cell
                    processed_row = [cell.strip() if cell else "" for cell in row]
                    table_data.append(processed_row)
                
                # Create a structured representation of the table
                if table_data:
                    # Get table position for context
                    bbox = table.bbox
                    tables.append({
                        'data': table_data,
                        'position': {
                            'x0': bbox.x0,
                            'y0': bbox.y0,
                            'x1': bbox.x1,
                            'y1': bbox.y1
                        }
                    })
        else:
            logger.debug("No tables found on page")
    except Exception as e:
        logger.warning(f"Error extracting tables: {str(e)}")
    return tables

def format_table_for_chunking(table_data):
    """Format table data into a text representation suitable for chunking."""
    if not table_data['data']:
        return ""
    
    # Get the maximum width of each column
    col_widths = []
    for row in table_data['data']:
        while len(col_widths) < len(row):
            col_widths.append(0)
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Format the table as text
    formatted_rows = []
    for row in table_data['data']:
        formatted_cells = []
        for i, cell in enumerate(row):
            formatted_cells.append(str(cell).ljust(col_widths[i]))
        formatted_rows.append(" | ".join(formatted_cells))
    
    # Add table context
    table_text = "Table Content:\n"
    table_text += "\n".join(formatted_rows)
    table_text += f"\nTable Position: x0={table_data['position']['x0']}, y0={table_data['position']['y0']}"
    
    return table_text

def process_pdf(pdf_file):
    """Process PDF file and return chunks with metadata."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(pdf_file.read())

        doc = fitz.open(tmp_path)
        chunks_with_metadata = []
        
        # Track document structure
        total_pages = len(doc)
        chapter_pattern = re.compile(r'^(?:Chapter|CHAPTER)\s+\d+', re.IGNORECASE)
        current_chapter = "Introduction"
        chapter_count = 0
        table_count = 0
        pages_with_tables = 0
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            
            # Detect chapter headers
            lines = text.split('\n')
            for line in lines[:5]:  # Check first few lines for chapter headers
                if chapter_pattern.match(line.strip()):
                    current_chapter = line.strip()
                    chapter_count += 1
                    break
            
            # Extract tables from the page
            tables = extract_table_data(page)
            if tables:
                pages_with_tables += 1
                logger.debug(f"Processing {len(tables)} tables on page {page_num + 1}")
            
            for table in tables:
                table_count += 1
                table_text = format_table_for_chunking(table)
                if table_text:
                    # Create a special chunk for the table
                    metadata = {
                        'is_table': True,
                        'table_number': table_count,
                        'page': page_num + 1,
                        'chapter': current_chapter,
                        'chapter_number': chapter_count,
                        'total_pages': total_pages,
                        'is_structural': True,
                        'page_number': page_num + 1,
                        'total_chapters': chapter_count,
                        'table_position': table['position']
                    }
                    chunks_with_metadata.append({
                        'text': table_text,
                        'metadata': metadata
                    })
            
            # Process regular text content
            sentences = sent_tokenize(text)
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                current_chunk.append(sentence)
                current_length += len(sentence)
                
                if current_length >= 1000:
                    chunk_text = ' '.join(current_chunk)
                    metadata = extract_metadata(chunk_text, page_num + 1)
                    # Add structural metadata
                    metadata.update({
                        'chapter': current_chapter,
                        'chapter_number': chapter_count,
                        'total_pages': total_pages,
                        'is_structural': True,
                        'page_number': page_num + 1,
                        'total_chapters': chapter_count,
                        'is_table': False
                    })
                    chunks_with_metadata.append({
                        'text': chunk_text,
                        'metadata': metadata
                    })
                    current_chunk = current_chunk[-3:]
                    current_length = sum(len(s) for s in current_chunk)
            
            # Process any remaining sentences in the last chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                metadata = extract_metadata(chunk_text, page_num + 1)
                # Add structural metadata
                metadata.update({
                    'chapter': current_chapter,
                    'chapter_number': chapter_count,
                    'total_pages': total_pages,
                    'is_structural': True,
                    'page_number': page_num + 1,
                    'total_chapters': chapter_count,
                    'is_table': False
                })
                chunks_with_metadata.append({
                    'text': chunk_text,
                    'metadata': metadata
                })

        # Add a special chunk for document structure
        structure_text = f"This document has {total_pages} pages and {chapter_count} chapters."
        if table_count > 0:
            structure_text += f" It contains {table_count} tables across {pages_with_tables} pages."
        else:
            structure_text += " No tables were found in this document."
            
        structure_chunk = {
            'text': structure_text,
            'metadata': {
                'is_structural': True,
                'total_pages': total_pages,
                'total_chapters': chapter_count,
                'total_tables': table_count,
                'pages_with_tables': pages_with_tables,
                'page': 0,
                'chapter': 'Document Structure',
                'is_table': False
            }
        }
        chunks_with_metadata.append(structure_chunk)

        logger.info(f"Document processing complete: {total_pages} pages, {chapter_count} chapters, {table_count} tables")
        doc.close()
        return chunks_with_metadata
    
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Could not delete temporary file {tmp_path}: {str(e)}")

def create_vector_store(chunks_with_metadata):
    """Create vector store from text chunks with metadata."""
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            chunk_size=1000
        )
        
        documents = []
        for chunk_idx, chunk in enumerate(chunks_with_metadata):
            # Enhance text with structural information for better retrieval
            enhanced_text = f"{chunk['text']} "
            if chunk['metadata'].get('is_structural'):
                enhanced_text += f"Document structure: {chunk['metadata'].get('chapter', '')} "
                enhanced_text += f"Page {chunk['metadata'].get('page_number', '')} of {chunk['metadata'].get('total_pages', '')} "
                enhanced_text += f"Chapter {chunk['metadata'].get('chapter_number', '')} of {chunk['metadata'].get('total_chapters', '')} "
            if chunk['metadata'].get('is_table'):
                enhanced_text += f"Table {chunk['metadata'].get('table_number', '')} "
            enhanced_text += f"{' '.join(chunk['metadata'].get('concepts', []))} {' '.join(chunk['metadata'].get('entities', []))}"
            
            doc = Document(
                page_content=enhanced_text,
                metadata=chunk['metadata']
            )
            documents.append(doc)
        
        if not documents:
            logger.warning("No documents were created from the PDF chunks. Vector store will be empty.")
            return None

        logger.debug(f"Creating vector store with {len(documents)} documents.")
        vector_store_instance = FAISS.from_documents(
            documents,
            embeddings
        )
        logger.debug("FAISS.from_documents call completed.")
        
        return vector_store_instance
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}", exc_info=True)
        raise

def setup_conversation_chain(vector_store_instance):
    """Set up the conversation chain with memory and enhanced retrieval."""
    if vector_store_instance is None:
        logger.error("Vector store is None, cannot set up conversation chain.")
        raise ValueError("Vector store cannot be None for setting up conversation chain.")

    # Use Gemini model for summarization/conversation
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro-preview-05-06", 
        temperature=0.7,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        convert_system_message_to_human=True
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Enhanced retriever configuration
    retriever = vector_store_instance.as_retriever(
        search_kwargs={
            'k': 8,  # Increased number of retrieved documents
            'fetch_k': 30,  # Increased fetch size
            'filter': None  # No filtering to ensure structural chunks are included
        }
    )
    
    prompt_template = PromptTemplate(
        template="""You are an AI assistant analyzing a document. Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        
        For questions about document structure (like number of chapters, pages, etc.), pay special attention to chunks marked as structural information.
        For content questions, use the semantic content and relationships to provide a comprehensive answer.
        
        Context: {context}
        
        Chat History:
        {chat_history}
        
        Question: {question}
        
        Answer:""",
        input_variables=["context", "chat_history", "question"]
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={
            'prompt': prompt_template,
            'document_variable_name': 'context'
        },
        verbose=True
    )
    
    return conversation_chain

@app.route('/')
def serve_index():
    logger.debug("Serving index.html")
    return send_from_directory('static', 'index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global conversation, vector_store, chat_history # Ensure global vector_store is updated
    
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
        chunks = process_pdf(file)
        logger.debug(f"Created {len(chunks)} chunks")
        
        if not chunks:
            logger.error("No chunks were created from the PDF.")
            return jsonify({'error': 'Could not extract text from PDF or PDF is empty.'}), 500

        # Assign to the global vector_store
        vector_store = create_vector_store(chunks) 
        test_results = vector_store.similarity_search("What is money ?");
        print(f"retrieved {len(test_results)} test documents");
        for doc in test_results:
            print(doc.page_content[:200])
        logger.debug("Create vector store call completed.")

        if vector_store is None:
            logger.error("Failed to create vector store.")
            return jsonify({'error': 'Failed to create vector store.'}), 500
        
        conversation = setup_conversation_chain(vector_store) # Pass the created store
        logger.debug("Set up conversation chain")
        
        chat_history = []
        
        return jsonify({'message': 'PDF processed successfully'}), 200
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    global conversation, chat_history # conversation uses the global vector_store implicitly
    
    logger.debug("Received question request")
    
    if not conversation:
        logger.error("No conversation initialized")
        return jsonify({'error': 'Please upload a PDF first'}), 400
    
    data = request.get_json()
    if not data or 'question' not in data:
        logger.error("No question in request")
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        question = data['question']
        logger.debug(f"Processing question: {question}")
        
        # The conversation chain will use the retriever (which has the vector_store)
        # to fetch context.
        response = conversation({"question": question})
        
        chat_history.append({
            'question': question,
            'answer': response['answer'],
            'sources': [
                {
                    # Ensure metadata and page number exist
                    'page': doc.metadata.get('page', 'N/A'), 
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
