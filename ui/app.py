import streamlit as st
import requests
import os

# Backend API URL
API_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Research Assistant", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for Sleek Light/Grey UI ---
st.markdown("""
    <style>
        /* Import Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        /* Global Theme overrides */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #F8F9FA;
            color: #1E293B;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Main container styling */
        .block-container {
            padding-top: 3rem !important;
            padding-bottom: 3rem !important;
            max-width: 1200px;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #FFFFFF;
            border-right: 1px solid #E2E8F0;
            box-shadow: 2px 0 10px rgba(0,0,0,0.02);
        }
        
        /* Cards & Answer Boxes */
        .st-emotion-cache-16txtl3 { /* Standard Streamlit container */
            padding: 2rem;
            border-radius: 12px;
            background: #FFFFFF;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            border: 1px solid #E2E8F0;
            margin-bottom: 1.5rem;
        }

        /* Titles and Headers */
        h1, h2, h3 {
            color: #0F172A;
            font-weight: 600 !important;
            letter-spacing: -0.02em;
        }
        
        h1 {
            font-size: 2.2rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* Buttons */
        .stButton > button {
            background-color: #0F172A;
            color: #FFFFFF;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1.5rem;
            font-weight: 500;
            transition: all 0.2s ease;
            box-shadow: 0 2px 4px rgba(15, 23, 42, 0.2);
            width: 100%;
        }
        .stButton > button:hover {
            background-color: #334155;
            box-shadow: 0 4px 6px rgba(15, 23, 42, 0.25);
            transform: translateY(-1px);
            color: white;
            border: none;
        }
        
        /* Text inputs */
        .stTextInput > div > div > input {
            border-radius: 8px;
            border: 1px solid #CBD5E1;
            padding: 0.75rem 1rem;
            background-color: #FFFFFF;
            color: #1E293B;
            font-size: 1rem;
            box-shadow: inset 0 1px 2px rgba(0,0,0,0.02);
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        .stTextInput > div > div > input:focus {
            border-color: #3B82F6;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }
        
        /* Subheaders in answers */
        .answer-label {
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748B;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        /* Citations pill */
        .citation-pill {
            display: inline-block;
            background-color: #F1F5F9;
            border: 1px solid #E2E8F0;
            color: #475569;
            padding: 4px 10px;
            border-radius: 16px;
            font-size: 0.8rem;
            margin-right: 6px;
            margin-bottom: 6px;
            font-weight: 500;
        }
        
        /* Warning Box */
        .stAlert {
            border-radius: 8px;
            border: none;
        }
    </style>
""", unsafe_allow_html=True)

# --- App Header ---
st.markdown("<h1>🔬 Sentinel: AI Research Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #64748B; font-size: 1.1rem; margin-bottom: 2rem;'>An intelligent, multi-document analysis engine powered by Hybrid Search and Cross-Encoder Re-ranking.</p>", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("<h2 style='font-size: 1.5rem; padding-top: 1rem;'>📄 Knowledge Base</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748B; font-size: 0.9rem;'>Upload PDFs to build your semantic index.</p>", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader("Drop documents here", type="pdf", accept_multiple_files=True, label_visibility="collapsed")
    
    if st.button("Initialize Documents"):
        if uploaded_files:
            with st.spinner("Extracting & Indexing..."):
                for file in uploaded_files:
                    files = {"file": (file.name, file, "application/pdf")}
                    try:
                        response = requests.post(f"{API_URL}/upload_pdf", files=files)
                        if response.status_code == 200:
                            data = response.json()
                            res = data.get("result", {})
                            if res.get("status") == "success":
                                st.success(f"Indexed: {file.name}")
                                with st.expander("View Auto-Summary"):
                                    st.write(res.get('summary', 'No summary available.'))
                            elif res.get("status") == "already processed":
                                st.info(f"Already Indexed: {file.name}")
                            else:
                                st.error(f"Error: {res.get('message')}")
                        else:
                            st.error(f"Failed to connect to backend for {file.name}.")
                    except Exception as e:
                        st.error(f"Connection error: {e}")
        else:
            st.warning("Please upload a PDF first.")
            
    st.markdown("---")
    st.markdown("<p style='font-size: 0.8rem; color: #94A3B8; text-align: center;'>Powered by FAISS, BM25 & FLAN-T5</p>", unsafe_allow_html=True)


# --- Main Interface ---
st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

# Search Bar
query = st.text_input("Query", placeholder="Ask a deep analytical question across your documents...", label_visibility="collapsed")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    search_clicked = st.button("Synthesize Answer")

if search_clicked:
    if query:
        with st.spinner("Analyzing cross-document context..."):
            try:
                response = requests.post(f"{API_URL}/query", json={"query": query})
                if response.status_code == 200:
                    data = response.json()
                    
                    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
                    
                    # Answer Container
                    st.markdown("<div class='st-emotion-cache-16txtl3'>", unsafe_allow_html=True)
                    st.markdown("<div class='answer-label'>Synthesized Response</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size: 1.1rem; line-height: 1.6; color: #334155;'>{data['answer']}</div>", unsafe_allow_html=True)
                    
                    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
                    st.markdown("<div class='answer-label'>Source Citations</div>", unsafe_allow_html=True)
                    
                    if data["citations"]:
                        citations_html = "".join([f"<span class='citation-pill'>{cit}</span>" for cit in data["citations"]])
                        st.markdown(citations_html, unsafe_allow_html=True)
                    else:
                        st.markdown("<span style='color: #94A3B8; font-size: 0.9rem;'>No exact citations found.</span>", unsafe_allow_html=True)
                        
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Contradictions
                    if data["contradictions"]:
                        st.error("⚠️ **Analytical Divergence Detected:** The system found conflicting facts across your sources.")
                        for conflict in data["contradictions"]:
                            with st.expander(f"Conflict Score: {conflict['score']:.2f}"):
                                st.markdown(f"**Source 1 {conflict['source_1']}:**\n> {conflict['text_1']}")
                                st.markdown(f"**Source 2 {conflict['source_2']}:**\n> {conflict['text_2']}")
                else:
                    st.error("❌ Failed to resolve query. Backend reported an error.")
            except Exception as e:
                st.error(f"❌ Connection error: {e}")
    else:
        st.warning("Please enter a research query.")
