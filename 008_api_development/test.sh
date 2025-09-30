#!/bin/bash

# Base URL
BASE_URL="http://localhost:8000"

# Temporary audio file
AUDIO_FILE="/tmp/my_audio.wav"

# Detect OS and record audio
echo "=== Recording 5 seconds of audio ==="
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux: use arecord
    arecord -f cd -t wav -d 5 -c 1 -r 16000 "$AUDIO_FILE"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS: use sox
    rec -c 1 -r 16000 "$AUDIO_FILE" trim 0 5
else
    echo "Unsupported OS for recording. Please record manually."
    exit 1
fi
echo "Audio recorded to $AUDIO_FILE"
echo -e "\n"

# 1. Health Check
echo "=== 1. Health Check ==="
curl -s -X GET "$BASE_URL/health"
echo -e "\n"

# 2. Create Draft
echo "=== 2. Create Draft ==="
DRAFT_RESPONSE=$(curl -s -X POST "$BASE_URL/voice-notes/draft" \
  -F "file=@$AUDIO_FILE")
echo "$DRAFT_RESPONSE"
DRAFT_ID=$(echo $DRAFT_RESPONSE | jq -r '.draftId')
echo "Draft ID: $DRAFT_ID"
echo -e "\n"

# 3. Update Draft
echo "=== 3. Update Draft ==="
curl -s -X PUT "$BASE_URL/voice-notes/draft/$DRAFT_ID?transcription=Updated transcription text&tags=tag1&tags=tag2"
echo -e "\n"

# 4. Finalize Draft
echo "=== 4. Finalize Draft ==="
NOTE_RESPONSE=$(curl -s -X POST "$BASE_URL/voice-notes?draftId=$DRAFT_ID")
echo "$NOTE_RESPONSE"
NOTE_ID=$(echo $NOTE_RESPONSE | jq -r '.id')
echo "Note ID: $NOTE_ID"
echo -e "\n"

# 5. List Notes
echo "=== 5. List Notes ==="
curl -s -X GET "$BASE_URL/voice-notes"
echo -e "\n"

# 5b. List Notes with search and tags
echo "=== 5b. List Notes with search and tags ==="
curl -s -X GET "$BASE_URL/voice-notes?search=Updated&tags=tag1"
echo -e "\n"

# 6. Get Specific Note
echo "=== 6. Get Specific Note ==="
curl -s -X GET "$BASE_URL/voice-notes/$NOTE_ID"
echo -e "\n"

echo "=== Script completed ==="

