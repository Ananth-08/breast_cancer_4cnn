import React, { useState } from "react";
import "./App.css";

function App() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    const formData = new FormData(e.target);

    const res = await fetch("http://localhost:8000/predict", {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    setResult(data);
    setLoading(false);
  };

  return (
    <div className="container">
      <header className="hero">
  <h1 className="hero-title">
    AI-Based Breast Cancer Analysis
  </h1>
  <p className="hero-subtitle">
    Deep learning–powered histopathology image assessment
    for benign and malignant tumor detection
  </p>
</header>



      <div className="main">
        <form className="card" onSubmit={handleSubmit}>
          <div className="upload-grid">
            <label className="upload-card">
              <input type="file" name="img40" hidden required />
              <span>📂 40X Image</span>
            </label>

            <label className="upload-card">
              <input type="file" name="img100" hidden required />
              <span>📂 100X Image</span>
            </label>

            <label className="upload-card">
              <input type="file" name="img200" hidden required />
              <span>📂 200X Image</span>
            </label>

            <label className="upload-card">
              <input type="file" name="img400" hidden required />
              <span>📂 400X Image</span>
            </label>
          </div>


          <button disabled={loading}>
            {loading ? "Analyzing..." : "Predict"}
          </button>
        </form>

        {result && (
          <div
            className={`result-card ${
              result.prediction === "MALIGNANT" ? "danger" : "safe"
            }`}
          >
            <h2>{result.prediction}</h2>
            <p>Confidence: {result.confidence_percent}%</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
