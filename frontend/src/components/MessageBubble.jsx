import "./MessageBubble.css";

export default function MessageBubble({ message }) {
  const { role, content, timestamp } = message;
  const isUser = role === "user";

  if (isUser) {
    return (
      <div className="message user">
        <div className="bubble user-bubble">
          <p>{content}</p>
        </div>
        <div className="avatar user-avatar">👤</div>
      </div>
    );
  }

  // Assistant message — content is the prediction object
  const { intent, confidence, entities, reasoning, latency_ms, model } = content;
  const confPct = (confidence * 100).toFixed(1);
  const confClass = confidence >= 0.85 ? "high" : confidence >= 0.6 ? "med" : "low";
  const entityEntries = Object.entries(entities || {});

  return (
    <div className="message assistant">
      <div className="avatar bot-avatar">🤖</div>
      <div className="bubble bot-bubble">
        {/* Intent badge */}
        <div className="result-header">
          <span className="intent-badge">{intent}</span>
          <span className={`confidence-badge ${confClass}`}>{confPct}%</span>
          {latency_ms > 0 && <span className="latency-badge">{latency_ms}ms</span>}
        </div>

        {/* Reasoning */}
        {reasoning && (
          <p className="reasoning">{reasoning}</p>
        )}

        {/* Entities */}
        {entityEntries.length > 0 && (
          <div className="entities-section">
            <span className="entities-label">Entities</span>
            <div className="entities-grid">
              {entityEntries.map(([key, val]) => (
                <div key={key} className="entity-chip">
                  <span className="entity-key">{key}</span>
                  <span className="entity-val">{val}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {model && <span className="model-tag">{model}</span>}
      </div>
    </div>
  );
}
