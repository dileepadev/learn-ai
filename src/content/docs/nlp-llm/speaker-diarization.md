---
title: Speaker Diarization - Answering Who Spoke When
description: Learn how speaker diarization segments audio by speaker identity, how it combines with transcription, and where it still struggles.
---

Speaker diarization answers "who spoke when" in an audio recording — segmenting a recording into time intervals and labeling each with a speaker identity, without necessarily knowing who those speakers actually are by name.

```text
[00:00-00:12] Speaker 1: "Let's start with the quarterly numbers."
[00:12-00:35] Speaker 2: "Sure, revenue was up eight percent."
[00:35-00:40] Speaker 1: "That's better than we forecasted."
```

## Diarization as a Distinct Task from Transcription

Diarization is a separate problem from automatic speech recognition (converting speech to text): a transcription model can produce accurate text without any notion of who said which words, and a diarization system can correctly segment speaker turns without transcribing any words at all. Production meeting-transcription and call-analysis systems combine both, aligning diarized speaker segments with transcribed text to produce a speaker-attributed transcript.

## Pipeline

A typical diarization pipeline runs voice activity detection to find speech segments (excluding silence and non-speech audio), extracts a speaker embedding for short segments across the recording using an embedding model trained similarly to the facial recognition embeddings discussed in [[facial-recognition-technology]] but for voice, then clusters these embeddings to group segments belonging to the same speaker without needing to know in advance how many speakers are present.

```text
audio -> voice activity detection -> speaker embeddings per segment -> clustering -> speaker labels
```

## Persistent Challenges

Overlapping speech, where two speakers talk simultaneously, breaks the assumption that each time segment belongs to exactly one speaker, and handling it well requires models specifically designed to detect and separate overlapping regions rather than simply assigning the whole segment to one speaker. Determining the correct number of speakers automatically (rather than requiring it as a fixed input) remains error-prone, especially in longer recordings where a speaker might be silent for long stretches, or where background voices briefly appear without being genuine participants.

## Practical Guidance

For meeting and call-recording products, evaluate diarization accuracy specifically on your target audio conditions (phone audio quality, number of typical speakers, amount of overlapping speech) rather than trusting benchmark numbers from clean, curated datasets. If speaker identity (not just distinguishing "speaker 1" from "speaker 2," but knowing their actual names) matters, pair diarization with a separate speaker identification step using enrolled voice profiles, since diarization alone only distinguishes speakers relative to each other within one recording.
