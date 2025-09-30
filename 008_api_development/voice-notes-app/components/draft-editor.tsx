"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { X, Plus, Loader2, Check } from "lucide-react"

interface Draft {
  draftId: string
  transcription: string
  tags: string[]
}

interface DraftEditorProps {
  draft: Draft
  backendUrl: string
  onComplete: () => void
}

export function DraftEditor({ draft, backendUrl, onComplete }: DraftEditorProps) {
  const [transcription, setTranscription] = useState(draft.transcription)
  const [tags, setTags] = useState<string[]>(draft.tags)
  const [newTag, setNewTag] = useState("")
  const [isSaving, setIsSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)

  const addTag = () => {
    const trimmedTag = newTag.trim()
    if (trimmedTag && !tags.includes(trimmedTag)) {
      setTags([...tags, trimmedTag])
      setNewTag("")
    }
  }

  const removeTag = (tagToRemove: string) => {
    setTags(tags.filter((tag) => tag !== tagToRemove))
  }

  const handleSave = async () => {
    setIsSaving(true)
    setError(null)

    try {
      const updateParams = new URLSearchParams()
      if (transcription) updateParams.append("transcription", transcription)
      tags.forEach((tag) => updateParams.append("tags", tag))

      const updateResponse = await fetch(
        `${backendUrl}/voice-notes/draft/${draft.draftId}?${updateParams.toString()}`,
        {
          method: "PUT",
        },
      )

      if (!updateResponse.ok) {
        throw new Error("Failed to update draft")
      }

      const saveResponse = await fetch(`${backendUrl}/voice-notes?draftId=${draft.draftId}`, {
        method: "POST",
      })

      if (!saveResponse.ok) {
        throw new Error("Failed to save note")
      }

      setSuccess(true)
      setTimeout(() => {
        onComplete()
      }, 1500)
    } catch (err) {
      setError("Failed to save note. Please try again.")
      console.error("Save error:", err)
    } finally {
      setIsSaving(false)
    }
  }

  if (success) {
    return (
      <div className="max-w-2xl mx-auto">
        <div className="glass rounded-3xl p-12 shadow-2xl">
          <div className="flex flex-col items-center gap-6 text-center">
            <div className="w-24 h-24 rounded-full gradient-accent shadow-2xl flex items-center justify-center relative overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-br from-white/20 to-transparent rounded-full" />
              <Check className="w-12 h-12 text-white relative z-10" />
            </div>
            <div>
              <h3 className="text-3xl font-bold bg-gradient-to-r from-primary via-accent to-secondary bg-clip-text text-transparent">
                Note Saved!
              </h3>
              <p className="text-base text-muted-foreground mt-2">Your voice note has been saved successfully</p>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div className="glass rounded-3xl p-8 shadow-2xl space-y-8">
        <div className="space-y-2">
          <h2 className="text-3xl font-bold bg-gradient-to-r from-primary via-accent to-secondary bg-clip-text text-transparent">
            Review & Edit
          </h2>
          <p className="text-base text-muted-foreground">Review the transcription and tags before saving your note</p>
        </div>

        <div className="space-y-3">
          <label htmlFor="transcription" className="text-sm font-semibold text-foreground">
            Transcription
          </label>
          <Textarea
            id="transcription"
            value={transcription}
            onChange={(e) => setTranscription(e.target.value)}
            rows={8}
            className="resize-none font-sans rounded-2xl border-2 bg-white/50 leading-relaxed"
            placeholder="Your transcription will appear here..."
          />
          <p className="text-xs text-muted-foreground font-medium">{transcription.length} characters</p>
        </div>

        <div className="space-y-4">
          <label className="text-sm font-semibold text-foreground">Tags</label>

          {tags.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {tags.map((tag, index) => (
                <Badge
                  key={tag}
                  className={`gap-1 pr-1 rounded-full text-white font-medium shadow-lg ${
                    index % 3 === 0 ? "gradient-primary" : index % 3 === 1 ? "gradient-secondary" : "gradient-accent"
                  }`}
                >
                  {tag}
                  <button
                    onClick={() => removeTag(tag)}
                    className="ml-1 hover:bg-white/20 rounded-full p-0.5 transition-colors"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </Badge>
              ))}
            </div>
          )}

          <div className="flex gap-2">
            <Input
              value={newTag}
              onChange={(e) => setNewTag(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && addTag()}
              placeholder="Add a tag..."
              className="flex-1 h-12 rounded-2xl border-2 bg-white/50"
            />
            <Button
              onClick={addTag}
              variant="outline"
              size="icon"
              disabled={!newTag.trim()}
              className="h-12 w-12 rounded-2xl border-2 hover:gradient-primary hover:text-white hover:border-transparent transition-all bg-transparent"
            >
              <Plus className="w-5 h-5" />
            </Button>
          </div>
        </div>

        {error && (
          <div className="text-sm text-destructive bg-destructive/10 border-2 border-destructive/30 rounded-2xl p-4">
            {error}
          </div>
        )}

        <div className="flex gap-4 pt-4">
          <Button
            onClick={onComplete}
            variant="outline"
            className="flex-1 h-12 rounded-2xl border-2 glass-dark hover:bg-muted transition-all bg-transparent"
            disabled={isSaving}
          >
            Cancel
          </Button>
          <Button
            onClick={handleSave}
            className="flex-1 h-12 rounded-2xl gradient-primary text-white font-semibold shadow-lg hover:shadow-xl transition-all hover:scale-[1.02]"
            disabled={isSaving || !transcription.trim()}
          >
            {isSaving ? (
              <>
                <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                Saving...
              </>
            ) : (
              "Save Note"
            )}
          </Button>
        </div>
      </div>
    </div>
  )
}
