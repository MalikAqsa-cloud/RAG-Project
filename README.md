🤖 AI Concept Explainer

An interactive RAG (Retrieval-Augmented Generation) app built with LangChain, Google Gemini, and Streamlit.
Ask AI-related questions and get accurate, context-aware answers from your custom knowledge base (ai_concepts.txt).

🚀 Quick Start
# 1. Clone repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key in .env
echo "GOOGLE_API_KEY=your_google_api_key_here" > .env

# 5. Run the app
streamlit run app.py

🖼️ Features

✅ RAG Pipeline with Chroma vector DB
✅ Google Gemini (gemini-pro) for intelligent responses
✅ Streamlit UI for simple Q&A

🧠 Tech Stack

LangChain • Google Gemini API • ChromaDB • Streamlit • dotenv
