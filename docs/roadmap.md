# Roadmap

## Released — v0.1.0

- [x] Temporal knowledge graph (S-P-O + timestamp + weight)
- [x] Pattern-based triple extraction (EN + ZH)
- [x] Multi-hop causal retrieval with 3 recall modes
- [x] Self-evolution: decay / prune / merge / conflict detection
- [x] SQLite persistence with WAL + schema migration
- [x] Hermes MemoryProvider ABC full implementation
- [x] Token budget compression + prefetch cache
- [x] 75 tests, zero external dependencies

## Planned — v0.2.0

- [ ] Async prefetch via asyncio background tasks
- [ ] LLM-assisted triple extraction (optional adapter)
- [ ] Spacy / stanza NER integration (optional)
- [ ] Memory export to Obsidian-compatible Markdown

## Planned — v0.3.0

- [ ] Neo4j backend adapter
- [ ] Distributed store for multi-agent setups
- [ ] REST API for memory graph inspection
- [ ] Web UI: interactive graph visualizer

## Long-term

- [ ] SAGE full paper architecture (RL-driven self-optimization)
- [ ] Cross-agent memory sharing
- [ ] Fine-tuned extraction model (distilled from GPT-4o)