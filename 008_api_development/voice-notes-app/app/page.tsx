"use client"

import { useState, useEffect } from "react"
import { BackendSetup } from "@/components/backend-setup"
import { RecordingInterface } from "@/components/recording-interface"
import { NotesList } from "@/components/notes-list"

export default function Home() {
  const [backendUrl, setBackendUrl] = useState<string | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [activeView, setActiveView] = useState<"record" | "list">("record")

  // Check if backend URL is stored in session
  useEffect(() => {
    const storedUrl = sessionStorage.getItem("backendUrl")
    if (storedUrl) {
      setBackendUrl(storedUrl)
      setIsConnected(true)
    }
  }, [])

  const handleBackendConnect = (url: string) => {
    sessionStorage.setItem("backendUrl", url)
    setBackendUrl(url)
    setIsConnected(true)
  }

  const handleDisconnect = () => {
    sessionStorage.removeItem("backendUrl")
    setBackendUrl(null)
    setIsConnected(false)
  }

  if (!isConnected || !backendUrl) {
    return <BackendSetup onConnect={handleBackendConnect} />
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-6">
            <h1 className="text-xl font-semibold text-foreground">Voice Notes</h1>
            <nav className="flex gap-1">
              <button
                onClick={() => setActiveView("record")}
                className={`px-4 py-2 rounded-md text-sm transition-colors ${
                  activeView === "record"
                    ? "bg-secondary text-foreground"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                Record
              </button>
              <button
                onClick={() => setActiveView("list")}
                className={`px-4 py-2 rounded-md text-sm transition-colors ${
                  activeView === "list" ? "bg-secondary text-foreground" : "text-muted-foreground hover:text-foreground"
                }`}
              >
                Notes
              </button>
            </nav>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-xs text-muted-foreground font-mono">{backendUrl}</span>
            <button
              onClick={handleDisconnect}
              className="text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              Disconnect
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {activeView === "record" ? (
          <RecordingInterface backendUrl={backendUrl} />
        ) : (
          <NotesList backendUrl={backendUrl} />
        )}
      </main>
    </div>
  )
}
