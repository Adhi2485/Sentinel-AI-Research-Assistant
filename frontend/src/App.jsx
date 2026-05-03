import { useState, useRef, useEffect } from 'react';
import './index.css';

const API_URL = "http://localhost:8000";

function App() {
  const [query, setQuery] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [loadingSearch, setLoadingSearch] = useState(false);
  
  const [uploadStatus, setUploadStatus] = useState("");
  const [loadingUpload, setLoadingUpload] = useState(false);
  const fileInputRef = useRef(null);
  
  // Telemetry state
  const [showTelemetry, setShowTelemetry] = useState(false);
  const [telemetry, setTelemetry] = useState(null);

  const fetchTelemetry = async () => {
    try {
      const response = await fetch(`${API_URL}/system_telemetry`);
      if (response.ok) {
        const data = await response.json();
        setTelemetry(data);
      }
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    if (showTelemetry) {
      fetchTelemetry();
    }
  }, [showTelemetry]);

  const handleClearMemory = async () => {
    if (!window.confirm("Are you sure you want to wipe the Vector Database and all Indexed Documents?")) return;
    try {
      const response = await fetch(`${API_URL}/clear_memory`, { method: "DELETE" });
      if (response.ok) {
        setUploadStatus("Database completely wiped.");
        setChatHistory([]);
        if (showTelemetry) fetchTelemetry();
      }
    } catch (e) {
      alert("Failed to clear memory.");
    }
  };

  const handleUpload = async (e) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setLoadingUpload(true);
    setUploadStatus("Uploading and indexing...");

    for (let i = 0; i < files.length; i++) {
      const formData = new FormData();
      formData.append("file", files[i]);

      try {
        const response = await fetch(`${API_URL}/upload_pdf`, {
          method: "POST",
          body: formData,
        });
        
        const data = await response.json();
        if (response.ok && data.result.status !== "error") {
          setUploadStatus(prev => prev + `\nIndexed: ${files[i].name}`);
          if (showTelemetry) fetchTelemetry();
        } else {
          setUploadStatus(prev => prev + `\nError: ${files[i].name}`);
        }
      } catch (error) {
        setUploadStatus(prev => prev + `\nFailed to connect.`);
      }
    }
    setLoadingUpload(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    const currentQuery = query;
    setQuery("");
    setLoadingSearch(true);

    try {
      const response = await fetch(`${API_URL}/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: currentQuery }),
      });
      
      if (response.ok) {
        const data = await response.json();
        setChatHistory(prev => [{ query: currentQuery, result: data }, ...prev]);
      } else {
        alert("Failed to resolve query.");
      }
    } catch (error) {
      alert("Connection error: " + error.message);
    }
    
    setLoadingSearch(false);
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <h2>📄 Knowledge Base</h2>
        <p>Upload PDFs to build your semantic index.</p>
        
        <div className="file-input-wrapper">
          <label className="file-input">
            {loadingUpload ? "Indexing..." : "Click to select PDFs"}
            <input 
              type="file" 
              multiple 
              accept=".pdf" 
              onChange={handleUpload}
              disabled={loadingUpload}
              ref={fileInputRef}
            />
          </label>
        </div>
        
        {uploadStatus && (
          <div className="status-box" style={{ whiteSpace: "pre-line", maxHeight: '200px', overflowY: 'auto' }}>
            {uploadStatus}
          </div>
        )}
        
        <div style={{ marginTop: '1rem' }}>
          <button className="btn-secondary" style={{ borderColor: '#FECACA', color: '#B91C1C' }} onClick={handleClearMemory}>
            🗑️ Clear Database Memory
          </button>
        </div>

        <div style={{ marginTop: 'auto', paddingTop: '2rem' }}>
          <button className="btn-secondary" onClick={() => setShowTelemetry(!showTelemetry)}>
            {showTelemetry ? "Hide System Telemetry" : "📊 View System Telemetry"}
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <div className="header">
          <h1>Sentinel: AI Research</h1>
          <p>An intelligent, multi-document analysis engine powered by Hybrid Search.</p>
        </div>

        {showTelemetry && telemetry && (
          <div className="telemetry-box">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3 style={{ margin: 0 }}>Backend Telemetry & Memory State</h3>
              <button className="btn-secondary" style={{ width: 'auto', padding: '0.2rem 0.5rem', fontSize: '0.8rem' }} onClick={fetchTelemetry}>🔄 Refresh</button>
            </div>
            <div className="telemetry-grid">
              <div className="tel-stat"><strong>Status:</strong> {telemetry.status}</div>
              <div className="tel-stat"><strong>Total Indexed Files:</strong> {telemetry.documents_indexed}</div>
              <div className="tel-stat"><strong>Total Chunks in RAM:</strong> {telemetry.total_chunks_in_memory}</div>
              <div className="tel-stat"><strong>FAISS Index Size:</strong> {telemetry.faiss_index_size}</div>
              <div className="tel-stat" style={{ gridColumn: '1 / -1' }}><strong>Document Names:</strong> {telemetry.document_names.join(', ') || 'None'}</div>
            </div>
          </div>
        )}

        <form className="search-container" onSubmit={handleSearch}>
          <input 
            type="text" 
            className="search-input"
            placeholder="Ask a deep analytical question across your documents..." 
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={loadingSearch}
          />
          <button type="submit" className="btn-primary" style={{ width: 'auto' }} disabled={loadingSearch || !query.trim()}>
            {loadingSearch ? "Synthesizing..." : "Synthesize Answer"}
          </button>
        </form>

        <div className="chat-history">
          {chatHistory.map((chat, index) => (
            <div key={index} className="chat-item">
              <div className="chat-query">
                <span className="query-icon">👤</span>
                <h3>{chat.query}</h3>
              </div>
              
              <div className="result-card">
                <div className="label">Synthesized Response</div>
                <div className="answer-text">
                  {chat.result.answer}
                </div>

                <div className="label">Source Citations</div>
                <div className="citations-container">
                  {chat.result.citations.length > 0 ? (
                    chat.result.citations.map((cit, idx) => (
                      <span key={idx} className="citation-pill">{cit}</span>
                    ))
                  ) : (
                    <span style={{ color: "#94A3B8" }}>No exact citations found.</span>
                  )}
                </div>
                
                <details className="raw-retrieval-details">
                  <summary>View Cross-Encoder Re-Ranking Scores (Transparency)</summary>
                  <div className="retrieval-list">
                    {chat.result.retrieved_chunks.map((chunk, idx) => (
                      <div key={idx} className="retrieval-item">
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                          <strong>Rank {idx + 1} | {chunk.source}</strong>
                          <span className="score-badge">Score: {chunk.score.toFixed(3)}</span>
                        </div>
                        <p style={{ fontSize: '0.85rem', color: '#475569', margin: 0 }}>"{chunk.text}"</p>
                      </div>
                    ))}
                  </div>
                </details>

                {chat.result.contradictions && chat.result.contradictions.length > 0 && (
                  <div className="contradiction-box">
                    <h4>⚠️ Analytical Divergence Detected</h4>
                    {chat.result.contradictions.map((conflict, idx) => (
                      <div key={idx} className="conflict-item">
                        <div className="conflict-source">Source 1: {conflict.source_1}</div>
                        <div className="conflict-quote">"{conflict.text_1}"</div>
                        <div className="conflict-source" style={{ marginTop: '0.5rem' }}>Source 2: {conflict.source_2}</div>
                        <div className="conflict-quote">"{conflict.text_2}"</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </main>
    </div>
  );
}

export default App;
