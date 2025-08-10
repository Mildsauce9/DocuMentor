document.addEventListener('DOMContentLoaded', () => {
    const pdfUpload = document.getElementById('pdf-upload');
    const uploadStatus = document.getElementById('upload-status');
    const chatSection = document.getElementById('chat-section');
    const questionInput = document.getElementById('question-input');
    const askButton = document.getElementById('ask-button');
    const chatHistory = document.getElementById('chat-history');

    // Handle file upload
    pdfUpload.addEventListener('change', handleFileUpload);
    
    // Handle drag and drop
    const dropZone = document.querySelector('.border-dashed');
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropZone.classList.add('drag-over');
    }

    function unhighlight(e) {
        dropZone.classList.remove('drag-over');
    }

    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            pdfUpload.files = files;
            handleFileUpload();
        }
    }

    async function handleFileUpload() {
        const file = pdfUpload.files[0];
        if (!file) {
            console.error('No file selected');
            return;
        }

        console.log('Uploading file:', file.name);

        // Show upload status
        uploadStatus.classList.remove('hidden');
        
        const formData = new FormData();
        formData.append('file', file);

        try {
            console.log('Sending upload request...');
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData,
                credentials: 'include'
            });

            console.log('Upload response status:', response.status);
            const data = await response.json();
            console.log('Upload response data:', data);
            
            if (response.ok) {
                // Hide upload section and show chat section
                document.getElementById('upload-section').classList.add('hidden');
                chatSection.classList.remove('hidden');
                uploadStatus.classList.add('hidden');
                
                // Show success message
                const successMessage = document.createElement('div');
                successMessage.className = 'text-green-600 text-center mb-4';
                successMessage.textContent = 'PDF processed successfully! You can now ask questions.';
                chatSection.insertBefore(successMessage, chatSection.firstChild);
            } else {
                throw new Error(data.error || 'Failed to process PDF');
            }
        } catch (error) {
            console.error('Upload error:', error);
            alert('Error uploading file: ' + error.message);
        } finally {
            uploadStatus.classList.add('hidden');
        }
    }

    // Handle asking questions
    async function askQuestion() {
        const question = questionInput.value.trim();
        if (!question) return;

        console.log('Asking question:', question);

        // Add question to chat
        addMessageToChat('question', question);
        questionInput.value = '';

        try {
            console.log('Sending question request...');
            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ question }),
                credentials: 'include'
            });

            console.log('Question response status:', response.status);
            const data = await response.json();
            console.log('Question response data:', data);
            
            if (response.ok) {
                addMessageToChat('answer', data.answer);
                
                // Add sources if available
                if (data.sources && data.sources.length > 0) {
                    const sourcesDiv = document.createElement('div');
                    sourcesDiv.className = 'text-sm text-gray-600 mt-2';
                    sourcesDiv.innerHTML = '<strong>Sources:</strong><br>' + 
                        data.sources.map(source => 
                            `Page ${source.page}: ${source.content}`
                        ).join('<br>');
                    chatHistory.lastElementChild.appendChild(sourcesDiv);
                }
            } else {
                throw new Error(data.error || 'Failed to get answer');
            }
        } catch (error) {
            console.error('Question error:', error);
            addMessageToChat('answer', 'Error: ' + error.message);
        }
    }

    // Add event listeners for asking questions
    askButton.addEventListener('click', askQuestion);
    questionInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            askQuestion();
        }
    });

    function addMessageToChat(type, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${type}`;
        messageDiv.textContent = content;
        chatHistory.appendChild(messageDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }
}); 