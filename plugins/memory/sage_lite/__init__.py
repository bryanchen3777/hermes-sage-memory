"""
Hermes Plugin 入口。
放置路徑：plugins/memory/sage_lite/__init__.py（複製或 symlink）
"""
from __future__ import annotations


def register(ctx) -> None:
    """Hermes Plugin 標準入口"""
    try:
        from sage_memory.adapter import SAGELiteProvider
        provider = SAGELiteProvider(top_k=5, max_hops=2, max_tokens=800)
        ctx.register_memory_provider(provider)
        print("[SAGE-lite] Memory provider registered")
    except Exception as e:
        print(f"[SAGE-lite] Failed to register: {e}")
        raise