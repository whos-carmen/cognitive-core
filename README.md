# Cognitive Core

A 1B local agentic router that knows what it doesn't know and delegates accordingly.

## Two Tracks

```
cognitive-core/
├── README.md
├── training/     → Build the model (AWS g7e.2xlarge)
└── router/       → Run the model (local 7900 XTX)
```

### [training/](training/README.md)

Merge GnLOLot + Luminia, fine-tune with SFT + DPO on AWS, deploy to HuggingFace.

```
Phase 0: EC2 setup
Phase 1: TIES merge
Phase 2: SFT (3 epochs)
Phase 3: DPO
Phase 4: Upload to HF
```

### [router/](router/README.md)

Serve the trained model locally, route between answering / tools / RAG / delegation,
persist memory across sessions, observe everything.

```
MiniCPM5-1B (router, port 8081)
  ├── Answer directly
  ├── Tool calls
  ├── RAG pipeline (port 8082)
  └── Memory (Mem0 / Chroma)
```
