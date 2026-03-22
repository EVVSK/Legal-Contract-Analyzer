"use client";

import { useState, useRef, useEffect } from "react";

interface Message {
  role: "user" | "ai";
  content: string;
  context?: string;
}

interface IngestResponse {
  status: string;
  message: string;
  details: {
    total_chunks: number;
    successful_upserts: number;
  };
}

interface AskResponse {
  answer: string;
  context_used: string;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [availableFiles, setAvailableFiles] = useState<string[]>([]);
  const [selectedFile, setSelectedFile] = useState<string>("All Contracts");

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleClearChat = () => {
    setMessages([]);
    setUploadStatus(null);
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setUploadStatus(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/ingest", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const data: IngestResponse = await response.json();
      setUploadStatus(
        `Uploaded "${file.name}" - ${data.details.total_chunks} chunks processed, ${data.details.successful_upserts} stored.`
      );
      // Add file to available files for filtering
      setAvailableFiles((prev) =>
        prev.includes(file.name) ? prev : [...prev, file.name]
      );
    } catch (error) {
      setUploadStatus(
        `Error: ${error instanceof Error ? error.message : "Upload failed"}`
      );
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const handleSubmit = async (e: { preventDefault: () => void }) => {
    e.preventDefault();
    const trimmedInput = input.trim();
    if (!trimmedInput || isThinking) return;

    const userMessage: Message = { role: "user", content: trimmedInput };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsThinking(true);

    try {
      const response = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: trimmedInput,
          file_name: selectedFile === "All Contracts" ? null : selectedFile,
        }),
      });

      if (!response.ok) {
        throw new Error(`Request failed: ${response.statusText}`);
      }

      const data: AskResponse = await response.json();
      const aiMessage: Message = {
        role: "ai",
        content: data.answer,
        context: data.context_used,
      };
      setMessages((prev) => [...prev, aiMessage]);
    } catch (error) {
      const errorMessage: Message = {
        role: "ai",
        content: `Error: ${error instanceof Error ? error.message : "Failed to get response"}`,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsThinking(false);
    }
  };

  return (
    <div className="flex h-screen bg-zinc-950">
      {/* Sidebar */}
      <aside className="w-80 border-r border-zinc-800/50 bg-zinc-900 flex flex-col">
        {/* Header */}
        <div className="p-6 border-b border-zinc-800/50">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center text-lg">
              ⚖️
            </div>
            <div>
              <h1 className="text-lg font-semibold text-zinc-100 tracking-tight">
                Legal RAG Engine
              </h1>
              <p className="text-xs text-zinc-500 font-medium">
                Enterprise Contract Analysis
              </p>
            </div>
          </div>
        </div>

        {/* Document Upload Section */}
        <div className="p-5 flex-1 overflow-y-auto">
          <div className="mb-2">
            <span className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">
              Document Control
            </span>
          </div>

          {/* Drag & Drop Upload Zone */}
          <label
            htmlFor="file-upload"
            className={`
              relative block w-full p-6 mt-3 rounded-xl border-2 border-dashed
              transition-all duration-200 cursor-pointer group
              ${isUploading
                ? "border-amber-500/50 bg-amber-500/5"
                : "border-zinc-700 hover:border-zinc-500 hover:bg-zinc-800/50"
              }
            `}
          >
            <div className="flex flex-col items-center text-center">
              {isUploading ? (
                <>
                  <svg
                    className="animate-spin h-8 w-8 text-amber-500 mb-3"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                    aria-hidden="true"
                  >
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="3"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    />
                  </svg>
                  <span className="text-sm font-medium text-amber-400">Processing document...</span>
                  <span className="text-xs text-zinc-500 mt-1">Chunking & embedding</span>
                </>
              ) : (
                <>
                  <div className="w-12 h-12 rounded-full bg-zinc-800 group-hover:bg-zinc-700 flex items-center justify-center mb-3 transition-colors">
                    <svg
                      className="w-6 h-6 text-zinc-400 group-hover:text-zinc-300 transition-colors"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                      aria-hidden="true"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={1.5}
                        d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                      />
                    </svg>
                  </div>
                  <span className="text-sm font-medium text-zinc-300">Drop contract here</span>
                  <span className="text-xs text-zinc-500 mt-1">or click to browse</span>
                  <span className="text-xs text-zinc-600 mt-2">PDF, TXT, DOC, DOCX</span>
                </>
              )}
            </div>
            <input
              id="file-upload"
              ref={fileInputRef}
              type="file"
              onChange={handleFileUpload}
              disabled={isUploading}
              accept=".pdf,.txt,.doc,.docx"
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
            />
          </label>

          {/* Upload Status */}
          {uploadStatus && !isUploading && (
            <div
              className={`mt-4 p-4 rounded-xl text-sm leading-relaxed ${
                uploadStatus.startsWith("Error")
                  ? "bg-red-500/10 text-red-400 ring-1 ring-red-500/20"
                  : "bg-emerald-500/10 text-emerald-400 ring-1 ring-emerald-500/20"
              }`}
            >
              <div className="flex items-start gap-2">
                {uploadStatus.startsWith("Error") ? (
                  <svg className="w-5 h-5 shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.28 7.22a.75.75 0 00-1.06 1.06L8.94 10l-1.72 1.72a.75.75 0 101.06 1.06L10 11.06l1.72 1.72a.75.75 0 101.06-1.06L11.06 10l1.72-1.72a.75.75 0 00-1.06-1.06L10 8.94 8.28 7.22z" clipRule="evenodd" />
                  </svg>
                ) : (
                  <svg className="w-5 h-5 shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.857-9.809a.75.75 0 00-1.214-.882l-3.483 4.79-1.88-1.88a.75.75 0 10-1.06 1.061l2.5 2.5a.75.75 0 001.137-.089l4-5.5z" clipRule="evenodd" />
                  </svg>
                )}
                <span>{uploadStatus}</span>
              </div>
            </div>
          )}
        </div>

        {/* New Conversation Button */}
        {messages.length > 0 && (
          <div className="px-5 pb-3">
            <button
              type="button"
              onClick={handleClearChat}
              className="w-full flex items-center justify-center gap-2 px-4 py-2.5
                rounded-xl text-sm font-medium
                text-zinc-400 hover:text-zinc-200
                bg-transparent hover:bg-zinc-800/50
                ring-1 ring-zinc-800 hover:ring-zinc-700
                transition-all duration-200"
            >
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M12 4v16m8-8H4"
                />
              </svg>
              New Conversation
            </button>
          </div>
        )}

        {/* Footer */}
        <div className="p-5 border-t border-zinc-800/50">
          <div className="flex items-center gap-2 text-xs text-zinc-600">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span>Powered by LlamaIndex + Groq</span>
          </div>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col bg-zinc-950">
        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center px-6">
              <div className="w-20 h-20 mb-6 rounded-2xl bg-zinc-900 ring-1 ring-zinc-800 flex items-center justify-center">
                <svg
                  className="w-10 h-10 text-zinc-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  aria-hidden="true"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1}
                    d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
                  />
                </svg>
              </div>
              <h2 className="text-xl font-semibold text-zinc-200 mb-2">
                Start a Conversation
              </h2>
              <p className="text-sm text-zinc-500 max-w-md leading-relaxed">
                Upload a legal contract using the sidebar, then ask questions about clauses, terms, obligations, or any specific details.
              </p>
              <div className="flex items-center gap-6 mt-8 text-xs text-zinc-600">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-lg bg-zinc-900 flex items-center justify-center">
                    <span>📄</span>
                  </div>
                  <span>Upload</span>
                </div>
                <svg className="w-4 h-4 text-zinc-700" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-lg bg-zinc-900 flex items-center justify-center">
                    <span>💬</span>
                  </div>
                  <span>Ask</span>
                </div>
                <svg className="w-4 h-4 text-zinc-700" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-lg bg-zinc-900 flex items-center justify-center">
                    <span>✨</span>
                  </div>
                  <span>Analyze</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto px-6 py-8 space-y-6">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex gap-4 ${message.role === "user" ? "flex-row-reverse" : "flex-row"}`}
                >
                  {/* Avatar */}
                  <div
                    className={`w-8 h-8 rounded-lg shrink-0 flex items-center justify-center text-sm ${
                      message.role === "user"
                        ? "bg-amber-500/20 text-amber-400"
                        : "bg-zinc-800 text-zinc-400"
                    }`}
                  >
                    {message.role === "user" ? "You" : "AI"}
                  </div>

                  {/* Message Bubble */}
                  <div
                    className={`max-w-[85%] ${
                      message.role === "user"
                        ? "bg-zinc-800 rounded-2xl rounded-tr-md px-4 py-3"
                        : "bg-transparent"
                    }`}
                  >
                    <p
                      className={`text-[15px] leading-relaxed whitespace-pre-wrap ${
                        message.role === "user" ? "text-zinc-100" : "text-zinc-300"
                      }`}
                    >
                      {message.content}
                    </p>

                    {/* Sources & Citations Accordion (AI messages only) */}
                    {message.role === "ai" && message.context && message.context !== "No context needed for general greetings." && (
                      <details className="mt-3 group">
                        <summary
                          className="flex items-center gap-2 px-3 py-2
                            text-xs font-medium text-zinc-500 hover:text-zinc-400
                            bg-zinc-900/50 hover:bg-zinc-900
                            rounded-lg border border-zinc-800 hover:border-zinc-700
                            cursor-pointer select-none
                            transition-all duration-200"
                        >
                          <svg
                            className="w-3.5 h-3.5 transition-transform duration-200 group-open:rotate-90"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                            aria-hidden="true"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M9 5l7 7-7 7"
                            />
                          </svg>
                          <svg
                            className="w-3.5 h-3.5"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                            aria-hidden="true"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={1.5}
                              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                            />
                          </svg>
                          Sources & Citations
                        </summary>
                        <div
                          className="mt-2 p-3
                            text-xs text-zinc-400 leading-relaxed
                            bg-zinc-900/50 rounded-lg
                            border border-zinc-800
                            max-h-64 overflow-y-auto"
                        >
                          <pre className="whitespace-pre-wrap font-mono text-[11px]">
                            {message.context}
                          </pre>
                        </div>
                      </details>
                    )}
                  </div>
                </div>
              ))}

              {/* Thinking Indicator */}
              {isThinking && (
                <div className="flex gap-4">
                  <div className="w-8 h-8 rounded-lg bg-zinc-800 shrink-0 flex items-center justify-center text-sm text-zinc-400">
                    AI
                  </div>
                  <div className="flex items-center gap-1 py-3">
                    <span className="w-2 h-2 bg-zinc-500 rounded-full animate-bounce" />
                    <span className="w-2 h-2 bg-zinc-500 rounded-full animate-bounce [animation-delay:150ms]" />
                    <span className="w-2 h-2 bg-zinc-500 rounded-full animate-bounce [animation-delay:300ms]" />
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input Bar */}
        <div className="p-6 bg-gradient-to-t from-zinc-950 via-zinc-950 to-transparent">
          <form onSubmit={handleSubmit} className="max-w-3xl mx-auto">
            {/* File Filter Dropdown */}
            <div className="flex items-center justify-center gap-2 mb-4">
              <label htmlFor="file-filter" className="text-xs text-zinc-500">
                Search in:
              </label>
              <select
                id="file-filter"
                value={selectedFile}
                onChange={(e) => setSelectedFile(e.target.value)}
                className="px-3 py-1.5 rounded-lg text-sm
                  bg-zinc-900 text-zinc-300
                  ring-1 ring-zinc-800
                  focus:outline-none focus:ring-2 focus:ring-amber-500/50
                  cursor-pointer transition-all duration-200"
              >
                <option value="All Contracts">All Contracts</option>
                {availableFiles.map((fileName) => (
                  <option key={fileName} value={fileName}>
                    {fileName}
                  </option>
                ))}
              </select>
              {selectedFile !== "All Contracts" && (
                <button
                  type="button"
                  onClick={() => setSelectedFile("All Contracts")}
                  className="text-xs text-zinc-500 hover:text-zinc-300 transition-colors"
                >
                  Clear
                </button>
              )}
            </div>

            <div className="relative flex items-center">
              <label htmlFor="chat-input" className="sr-only">
                Ask a question
              </label>
              <input
                id="chat-input"
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about your contract..."
                disabled={isThinking}
                className="w-full px-5 py-4 pr-14 rounded-2xl
                  bg-zinc-900 text-zinc-100
                  placeholder-zinc-500
                  ring-1 ring-zinc-800
                  focus:outline-none focus:ring-2 focus:ring-amber-500/50
                  disabled:opacity-50 disabled:cursor-not-allowed
                  transition-all duration-200"
              />
              <button
                type="submit"
                disabled={isThinking || !input.trim()}
                className="absolute right-2 w-10 h-10 rounded-xl
                  bg-amber-500 hover:bg-amber-400
                  disabled:bg-zinc-700 disabled:cursor-not-allowed
                  flex items-center justify-center
                  transition-colors duration-200"
                aria-label="Send message"
              >
                <svg
                  className="w-5 h-5 text-zinc-900 disabled:text-zinc-500"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  aria-hidden="true"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M5 12h14m-7-7l7 7-7 7"
                  />
                </svg>
              </button>
            </div>
            <p className="text-center text-xs text-zinc-600 mt-3">
              AI responses are generated from your uploaded documents
            </p>
          </form>
        </div>
      </main>
    </div>
  );
}
