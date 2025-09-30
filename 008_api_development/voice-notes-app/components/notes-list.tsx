"use client"

import { useState, useEffect } from "react"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Search, Loader2 } from "lucide-react"
import { NoteDetail } from "@/components/note-detail"

interface VoiceNote {
  id: string
  transcription: string
  tags: string[]
  createdAt: number // Unix timestamp from backend
}

interface NotesListProps {
  backendUrl: string
}

export function NotesList({ backendUrl }: NotesListProps) {
  const [notes, setNotes] = useState<VoiceNote[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState("")
  const [selectedTags, setSelectedTags] = useState<string[]>([])
  const [allTags, setAllTags] = useState<string[]>([])
  const [selectedNote, setSelectedNote] = useState<VoiceNote | null>(null)

  useEffect(() => {
    fetchNotes()
  }, [searchQuery, selectedTags])

  const fetchNotes = async () => {
    setIsLoading(true)
    setError(null)

    try {
      const params = new URLSearchParams()
      if (searchQuery) params.append("search", searchQuery)
      if (selectedTags.length > 0) params.append("tags", selectedTags.join(","))

      const response = await fetch(`${backendUrl}/voice-notes?${params.toString()}`)

      if (!response.ok) {
        throw new Error("Failed to fetch notes")
      }

      const data = await response.json()
      setNotes(data)

      // Extract all unique tags
      const tags = new Set<string>()
      data.forEach((note: VoiceNote) => {
        note.tags.forEach((tag) => tags.add(tag))
      })
      setAllTags(Array.from(tags).sort())
    } catch (err) {
      setError("Failed to load notes. Please try again.")
      console.error("Fetch error:", err)
    } finally {
      setIsLoading(false)
    }
  }

  const toggleTag = (tag: string) => {
    setSelectedTags((prev) => (prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag]))
  }

  const clearFilters = () => {
    setSearchQuery("")
    setSelectedTags([])
  }

  const formatDate = (timestamp: number) => {
    const date = new Date(timestamp * 1000)
    return new Intl.DateTimeFormat("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "numeric",
      minute: "2-digit",
    }).format(date)
  }

  if (selectedNote) {
    return <NoteDetail note={selectedNote} backendUrl={backendUrl} onClose={() => setSelectedNote(null)} />
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Search and Filters */}
      <div className="space-y-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search notes..."
            className="pl-10"
          />
        </div>

        {/* Tag Filters */}
        {allTags.length > 0 && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-foreground">Filter by tags</label>
              {(searchQuery || selectedTags.length > 0) && (
                <Button onClick={clearFilters} variant="ghost" size="sm" className="h-auto py-1 px-2 text-xs">
                  Clear filters
                </Button>
              )}
            </div>
            <div className="flex flex-wrap gap-2">
              {allTags.map((tag) => (
                <Badge
                  key={tag}
                  variant={selectedTags.includes(tag) ? "default" : "outline"}
                  className="cursor-pointer hover:bg-primary/80 transition-colors"
                  onClick={() => toggleTag(tag)}
                >
                  {tag}
                </Badge>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Notes List */}
      {isLoading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 text-muted-foreground animate-spin" />
        </div>
      ) : error ? (
        <div className="text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-md p-4">
          {error}
        </div>
      ) : notes.length === 0 ? (
        <div className="text-center py-12 space-y-2">
          <p className="text-muted-foreground">
            {searchQuery || selectedTags.length > 0 ? "No notes found matching your filters" : "No voice notes yet"}
          </p>
          <p className="text-sm text-muted-foreground">
            {searchQuery || selectedTags.length > 0
              ? "Try adjusting your search or filters"
              : "Start recording to create your first note"}
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {notes.map((note) => (
            <div
              key={note.id}
              className="bg-card border border-border rounded-lg p-4 hover:border-primary/50 transition-colors cursor-pointer"
              onClick={() => setSelectedNote(note)}
            >
              <div className="space-y-3">
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-foreground line-clamp-2 leading-relaxed">{note.transcription}</p>
                  </div>
                </div>

                {/* Footer */}
                <div className="flex items-center justify-between gap-4">
                  <div className="flex flex-wrap gap-1.5">
                    {note.tags.map((tag) => (
                      <Badge key={tag} variant="secondary" className="text-xs">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                  <span className="text-xs text-muted-foreground whitespace-nowrap">{formatDate(note.createdAt)}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
