"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Loader2, Sparkles } from "lucide-react"

interface BackendSetupProps {
  onConnect: (url: string) => void
}

export function BackendSetup({ onConnect }: BackendSetupProps) {
  const [url, setUrl] = useState("")
  const [isVerifying, setIsVerifying] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleConnect = async () => {
    setError(null)
    setIsVerifying(true)

    try {
      // Ensure URL doesn't end with slash
      const cleanUrl = url.trim().replace(/\/$/, "")

      // Verify backend is reachable
      const response = await fetch(`${cleanUrl}/health`)

      if (!response.ok) {
        throw new Error("Backend health check failed")
      }

      onConnect(cleanUrl)
    } catch (err) {
      setError("Failed to connect to backend. Please check the URL and try again.")
      console.error("Backend connection error:", err)
    } finally {
      setIsVerifying(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center p-4">
      <div className="w-full max-w-md space-y-8">
        <div className="text-center space-y-3">
          <div className="flex items-center justify-center gap-2">
            <div className="w-12 h-12 rounded-2xl gradient-primary shadow-lg flex items-center justify-center">
              <Sparkles className="w-6 h-6 text-white" />
            </div>
          </div>
          <h1 className="text-5xl font-bold bg-gradient-to-r from-primary via-accent to-secondary bg-clip-text text-transparent">
            Voice Notes
          </h1>
          <p className="text-muted-foreground text-lg">Connect to your backend to get started</p>
        </div>

        <div className="glass rounded-3xl p-8 shadow-2xl space-y-6">
          <div className="space-y-3">
            <label htmlFor="backend-url" className="text-sm font-semibold text-foreground">
              Backend URL
            </label>
            <Input
              id="backend-url"
              type="url"
              placeholder="https://your-backend.com"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleConnect()}
              className="font-mono text-sm h-12 rounded-2xl border-2 bg-white/50"
            />
          </div>

          {error && (
            <div className="text-sm text-destructive bg-destructive/10 border-2 border-destructive/30 rounded-2xl p-4">
              {error}
            </div>
          )}

          <Button
            onClick={handleConnect}
            disabled={!url || isVerifying}
            className="w-full h-12 rounded-2xl gradient-primary text-white font-semibold shadow-lg hover:shadow-xl transition-all hover:scale-[1.02]"
          >
            {isVerifying ? (
              <>
                <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                Verifying...
              </>
            ) : (
              "Connect"
            )}
          </Button>
        </div>

        <div className="text-center text-sm text-muted-foreground glass-dark rounded-2xl p-4">
          <p>The app will verify the backend is reachable using GET /health</p>
        </div>
      </div>
    </div>
  )
}
