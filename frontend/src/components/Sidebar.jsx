import "./Sidebar.css";

export default function Sidebar({
  conversations,
  activeId,
  onSelect,
  onNew,
  onDelete,
  open,
  onToggle,
  health,
}) {
  return (
    <aside className={`sidebar ${open ? "open" : "closed"}`}>
      <div className="sidebar-header">
        <h1 className="sidebar-logo">
          <span className="logo-icon">🤖</span> BotTrainer
        </h1>
        <button className="sidebar-toggle" onClick={onToggle} title="Close sidebar">
          ✕
        </button>
      </div>

      <button className="new-chat-btn" onClick={onNew}>
        <span>＋</span> New Chat
      </button>

      <div className="conversations-list">
        {conversations.map((c) => (
          <div
            key={c.id}
            className={`conv-item ${c.id === activeId ? "active" : ""}`}
            onClick={() => onSelect(c.id)}
          >
            <span className="conv-icon">💬</span>
            <span className="conv-title">{c.title}</span>
            <button
              className="conv-delete"
              onClick={(e) => {
                e.stopPropagation();
                onDelete(c.id);
              }}
              title="Delete"
            >
              🗑
            </button>
          </div>
        ))}
      </div>

      <div className="sidebar-footer">
        <div className={`health-badge ${health ? "online" : "offline"}`}>
          <span className="health-dot" />
          {health ? `${health.intents_loaded} intents · ${health.model}` : "API offline"}
        </div>
      </div>
    </aside>
  );
}
