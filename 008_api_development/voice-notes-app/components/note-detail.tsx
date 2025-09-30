"use client"

import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ArrowLeft, Volume2 } from "lucide-react"

interface VoiceNote {
  id: string
  transcription: string
  tags: string[]
  createdAt: number
}

interface NoteDetailProps {
  note: VoiceNote
  backendUrl: string
  onClose: () => void
}

export function NoteDetail({ note, onClose }: NoteDetailProps) {
  const formatDate = (timestamp: number) => {
    const date = new Date(timestamp * 1000)
    return new Intl.DateTimeFormat("en-US", {
      weekday: "long",
      month: "long",
      day: "numeric",
      year: "numeric",
      hour: "numeric",
      minute: "2-digit",
    }).format(date)
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div className="space-y-6">
        {/* Back Button */}
        <Button onClick={onClose} variant="ghost" className="gap-2 -ml-2">
          <ArrowLeft className="w-4 h-4" />
          Back to notes
        </Button>

        {/* Note Card */}
        <div className="bg-card border border-border rounded-lg p-6 space-y-6">
          {/* Header */}
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-muted-foreground">
              <Volume2 className="w-4 h-4" />
              <span className="text-sm">{formatDate(note.createdAt)}</span>
            </div>
          </div>

          {/* Tags */}
          {note.tags.length > 0 && (
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Tags</label>
              <div className="flex flex-wrap gap-2">
                {note.tags.map((tag) => (
                  <Badge key={tag} variant="secondary">
                    {tag}
                  </Badge>
                ))}
              </div>
            </div>
          )}

          {/* Transcription */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-foreground">Transcription</label>
            <div className="bg-secondary rounded-lg p-4">
              <p className="text-sm text-foreground leading-relaxed whitespace-pre-wrap">{note.transcription}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
