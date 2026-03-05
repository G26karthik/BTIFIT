import { useState, useRef, useEffect } from "react";
import Sidebar from "./components/Sidebar";
import ChatArea from "./components/ChatArea";
import "./App.css";

const API = "http://localhost:8000/api";

export default function App() {
  const [conversations, setConversations] = useState([
    { id: 1, title: "New chat", messages: [] },
  ]);
  const [activeId, setActiveId] = useState(1);
  const [loading, setLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [health, setHealth] = useState(null);
  const nextId = useRef(2);

  const active = conversations.find((c) => c.id === activeId);

  useEffect(() => {
    fetch(`${API}/health`)
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  const newChat = () => {
    const id = nextId.current++;
    setConversations((prev) => [
      ...prev,
      { id, title: "New chat", messages: [] },
    ]);
    setActiveId(id);
  };

  const deleteChat = (id) => {
    setConversations((prev) => {
      const filtered = prev.filter((c) => c.id !== id);
      if (filtered.length === 0) {
        const newId = nextId.current++;
        setActiveId(newId);
        return [{ id: newId, title: "New chat", messages: [] }];
      }
      if (activeId === id) setActiveId(filtered[0].id);
      return filtered;
    });
  };

  const sendMessage = async (text) => {
    const userMsg = {
      role: "user",
      content: text,
      timestamp: new Date().toISOString(),
    };

    setConversations((prev) =>
      prev.map((c) =>
        c.id === activeId
          ? {
              ...c,
              title: c.messages.length === 0 ? text.slice(0, 30) : c.title,
              messages: [...c.messages, userMsg],
            }
          : c
      )
    );

    setLoading(true);
    try {
      const res = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });
      const data = await res.json();
      const botMsg = {
        role: "assistant",
        content: data,
        timestamp: data.timestamp,
      };
      setConversations((prev) =>
        prev.map((c) =>
          c.id === activeId
            ? { ...c, messages: [...c.messages, botMsg] }
            : c
        )
      );
    } catch (err) {
      const errMsg = {
        role: "assistant",
        content: {
          intent: "error",
          confidence: 0,
          entities: {},
          reasoning: err.message,
          latency_ms: 0,
          model: "",
        },
        timestamp: new Date().toISOString(),
      };
      setConversations((prev) =>
        prev.map((c) =>
          c.id === activeId
            ? { ...c, messages: [...c.messages, errMsg] }
            : c
        )
      );
    }
    setLoading(false);
  };

  return (
    <div className="app">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        onSelect={setActiveId}
        onNew={newChat}
        onDelete={deleteChat}
        open={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
        health={health}
      />
      <ChatArea
        messages={active?.messages || []}
        onSend={sendMessage}
        loading={loading}
        sidebarOpen={sidebarOpen}
        onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
      />
    </div>
  );
}
