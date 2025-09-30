"use client";

import { useState, useRef } from "react";
import { Mic, Square, Loader2 } from "lucide-react";
import { DraftEditor } from "@/components/draft-editor";

interface RecordingInterfaceProps {
  backendUrl: string;
}

interface Draft {
  draftId: string;
  transcription: string;
  tags: string[];
}

export function RecordingInterface({ backendUrl }: RecordingInterfaceProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const startRecording = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: mediaRecorder.mimeType,
        });

        await uploadDraft(audioBlob);

        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      setError(
        "Failed to access microphone. Please grant permission and try again.",
      );
      console.error("Recording error:", err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const uploadDraft = async (audioBlob: Blob) => {
    setIsProcessing(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", audioBlob, "recording.webm");

      const response = await fetch(`${backendUrl}/voice-notes/draft`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to upload draft");
      }

      const data = await response.json();
      setDraft({
        draftId: data.draftId,
        transcription: data.transcription,
        tags: data.tags || [],
      });
    } catch (err) {
      setError("Failed to process recording. Please try again.");
      console.error("Upload error:", err);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDraftComplete = () => {
    setDraft(null);
  };

  if (draft) {
    return (
      <DraftEditor
        draft={draft}
        backendUrl={backendUrl}
        onComplete={handleDraftComplete}
      />
    );
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div className="glass rounded-3xl p-8 shadow-2xl space-y-8">
        <div className="text-center space-y-3">
          <h2 className="text-3xl font-bold bg-gradient-to-r from-primary via-accent to-secondary bg-clip-text text-transparent">
            Record a Voice Note
          </h2>
          <p className="text-base text-muted-foreground">
            {isRecording
              ? "Recording in progress... Click stop when finished"
              : "Click the microphone to start recording"}
          </p>
        </div>

        <div className="flex justify-center py-12">
          {isProcessing ? (
            <div className="flex flex-col items-center gap-6">
              <div className="w-32 h-32 rounded-full glass-dark shadow-2xl flex items-center justify-center">
                <Loader2 className="w-16 h-16 text-primary animate-spin" />
              </div>
              <p className="text-base text-muted-foreground font-medium">
                Processing recording...
              </p>
            </div>
          ) : isRecording ? (
            <button
              onClick={stopRecording}
              className="w-32 h-32 rounded-full gradient-secondary shadow-2xl hover:shadow-[0_20px_60px_rgba(236,72,153,0.5)] transition-all hover:scale-105 flex items-center justify-center group relative overflow-hidden"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-white/20 to-transparent rounded-full" />
              <Square className="w-12 h-12 text-white fill-current relative z-10" />
            </button>
          ) : (
            <button
              onClick={startRecording}
              className="w-32 h-32 rounded-full gradient-primary shadow-2xl hover:shadow-[0_20px_60px_rgba(99,102,241,0.5)] transition-all hover:scale-105 flex items-center justify-center group relative overflow-hidden"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-white/20 to-transparent rounded-full" />
              <Mic className="w-14 h-14 text-white relative z-10" />
            </button>
          )}
        </div>

        {isRecording && (
          <div className="flex items-center justify-center gap-3">
            <div className="relative">
              <div className="w-4 h-4 rounded-full bg-destructive animate-pulse" />
              <div className="absolute inset-0 w-4 h-4 rounded-full bg-destructive animate-ping" />
            </div>
            <span className="text-base text-muted-foreground font-medium">
              Recording...
            </span>
          </div>
        )}

        {error && (
          <div className="text-sm text-destructive bg-destructive/10 border-2 border-destructive/30 rounded-2xl p-4">
            {error}
          </div>
        )}

        <div className="glass-dark rounded-2xl p-6 space-y-4">
          <h3 className="text-base font-semibold text-foreground">
            How it works:
          </h3>
          <ol className="text-sm text-muted-foreground space-y-3 list-decimal list-inside leading-relaxed">
            <li>Click the microphone button to start recording</li>
            <li>Speak your voice note clearly</li>
            <li>Click the stop button when finished</li>
            <li>Review and edit the transcription and tags</li>
            <li>Save your note to the collection</li>
          </ol>
        </div>
      </div>
    </div>
  );
}
