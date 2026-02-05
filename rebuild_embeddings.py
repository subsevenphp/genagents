import argparse
import json
from pathlib import Path

from simulation_engine.gpt_structure import get_text_embedding


def _load_nodes(agent_folder: Path) -> list[dict]:
  nodes_path = agent_folder / "memory_stream" / "nodes.json"
  with nodes_path.open("r") as f:
    return json.load(f)


def _write_embeddings(agent_folder: Path, embeddings: dict) -> None:
  emb_path = agent_folder / "memory_stream" / "embeddings.json"
  backup = emb_path.with_suffix(".json.bak")
  if emb_path.exists():
    backup.write_text(emb_path.read_text())
  emb_path.write_text(json.dumps(embeddings))


def rebuild_agent_embeddings(agent_folder: Path) -> None:
  nodes = _load_nodes(agent_folder)
  embeddings = {}
  for node in nodes:
    content = node.get("content", "")
    if not isinstance(content, str) or not content.strip():
      continue
    if content in embeddings:
      continue
    embeddings[content] = get_text_embedding(content)
  _write_embeddings(agent_folder, embeddings)


def main():
  parser = argparse.ArgumentParser(
    description="Rebuild memory_stream embeddings for an agent using the current embedding backend."
  )
  parser.add_argument("--agent-folder", required=True,
                      help="Path to an agent folder containing memory_stream/nodes.json")
  args = parser.parse_args()
  agent_folder = Path(args.agent_folder).expanduser().resolve()
  rebuild_agent_embeddings(agent_folder)
  print(f"Rebuilt embeddings for {agent_folder}")


if __name__ == "__main__":
  main()
