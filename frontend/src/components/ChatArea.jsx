import { useState, useRef, useEffect } from "react";
import MessageBubble from "./MessageBubble";
import "./ChatArea.css";

export default function ChatArea({ messages, onSend, loading, sidebarOpen, onToggleSidebar }) {
  const [input, setInput] = useState("");
  const bottomRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleSubmit = (e) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || loading) return;
    setInput("");
    onSend(text);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <main className="chat-area">
      {/* Top bar */}
      <div className="chat-topbar">
        {!sidebarOpen && (
          <button className="menu-btn" onClick={onToggleSidebar} title="Open sidebar">
            ☰
          </button>
        )}
        <h2 className="topbar-title">BotTrainer NLU</h2>
        <span className="topbar-badge">Gemini 2.5 Flash</span>
      </div>

      {/* Messages */}
      <div className="messages-container">
        {messages.length === 0 && (
          <div className="welcome">
            <div className="welcome-icon">🧠</div>
            <h2>Welcome to BotTrainer</h2>
            <p>Type a message to classify its intent and extract entities using AI.</p>
            <div className="suggestions">
              {["What's my account balance?", "Book a flight to Mumbai", "What's the weather in Tokyo?", "Transfer $500 to savings"].map(
                (s) => (
                  <button key={s} className="suggestion-chip" onClick={() => { setInput(s); inputRef.current?.focus(); }}>
                    {s}
                  </button>
                )
              )}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <MessageBubble key={i} message={msg} />
        ))}

        {loading && (
          <div className="message assistant">
            <div className="avatar bot-avatar">🤖</div>
            <div className="bubble bot-bubble">
              <div className="typing-indicator">
                <span /><span /><span />
              </div>
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <form className="input-area" onSubmit={handleSubmit}>
        <div className="input-wrapper">
          <textarea
            ref={inputRef}
            className="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type your message…"
            rows={1}
            disabled={loading}
          />
          <button
            type="submit"
            className={`send-btn ${input.trim() && !loading ? "active" : ""}`}
            disabled={!input.trim() || loading}
          >
            ➤
          </button>
        </div>
        <p className="input-hint">BotTrainer classifies intent & extracts entities from your text</p>
      </form>
    </main>
  );
}
