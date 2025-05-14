# DocuMentor - Smart PDF Analyst

DocuMentor is an intelligent PDF analysis tool that uses RAG (Retrieval-Augmented Generation) to help you understand and interact with your PDF documents through natural language.

## Features

- 📄 PDF Processing: Upload and process any PDF document
- 💬 Natural Language Q&A: Ask questions about your document in plain English
- 🧠 Contextual Memory: Maintains conversation context for follow-up questions
- 🔍 Source-Aware Answers: Get answers with references to the original content
- 📊 Insight Generation: Automatically extracts key topics and entities

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DocuMentor.git
cd DocuMentor
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

5. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Open your browser and navigate to `http://localhost:8501`
2. Upload a PDF document using the file uploader
3. Wait for the document to be processed
4. Start asking questions about the content
5. View the chat history and insights

## Requirements

- Python 3.8+
- OpenAI API key
- Internet connection for API calls

## License

MIT License 