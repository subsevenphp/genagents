import argparse
import json
from pathlib import Path

from genagents.genagents import GenerativeAgent


def _load_config(path: Path) -> dict:
  with path.open("r") as f:
    return json.load(f)


def _resolve_agent_path(base_dir: Path, raw_path: str) -> str:
  p = Path(raw_path)
  if p.is_absolute():
    return str(p)
  return str((base_dir / p).resolve())


def _get_relationship(config: dict, speaker_key: str, listener_key: str) -> str:
  rels = config.get("relationships", {})
  if isinstance(rels, dict):
    key = f"{speaker_key}->{listener_key}"
    if key in rels:
      return rels[key]
    if speaker_key in rels and isinstance(rels[speaker_key], dict):
      return rels[speaker_key].get(listener_key, "")
  return ""


def _agent_name(key: str, agent: GenerativeAgent, prefer_key: bool) -> str:
  if prefer_key:
    return key
  name = agent.get_fullname()
  return name if name else key


def _build_context(topic: str,
                   speaker_key: str,
                   listener_key: str,
                   speaker_name: str,
                   listener_name: str,
                   relationship: str,
                   opening: str,
                   max_words: int | None,
                   strict: bool) -> str:
  lines = [
    f"Topic: {topic}",
    f"Your name: {speaker_name}",
    f"Your partner: {listener_name}"
  ]
  if relationship:
    lines.append(f"Relationship: {relationship}")
  if opening:
    lines.append(f"Opening instruction: {opening}")
  if max_words:
    lines.append(f"Keep your response under {max_words} words.")
  if strict:
    lines.append(
      "Hard rules: Stay strictly on topic and remain fully consistent with the relationship."
    )
    lines.append(
      "If a response would violate the topic or relationship, revise it until it complies."
    )
  else:
    lines.append("Stay on topic and be consistent with the relationship.")
  return "\n".join(lines)


def run_conversation(config_path: Path,
                     topic: str,
                     agent_keys: list[str],
                     turns: int,
                     start_key: str | None,
                     opening: str,
                     max_words: int | None,
                     out_path: Path | None,
                     display_keys: bool,
                     strict: bool) -> None:
  config = _load_config(config_path)
  base_dir = config_path.parent

  agents_cfg = config.get("agents", {})
  if not isinstance(agents_cfg, dict) or not agents_cfg:
    raise ValueError("Config must include an 'agents' object with at least two entries.")

  if len(agent_keys) != 2:
    raise ValueError("Please provide exactly two agent keys via --agents.")

  for key in agent_keys:
    if key not in agents_cfg:
      raise ValueError(f"Unknown agent key: {key}")

  agents = {}
  for key in agent_keys:
    agent_path = _resolve_agent_path(base_dir, agents_cfg[key])
    agents[key] = GenerativeAgent(agent_path)

  a_key, b_key = agent_keys
  if start_key:
    if start_key not in agent_keys:
      raise ValueError("--start must be one of the two agent keys.")
    order = [start_key, b_key if start_key == a_key else a_key]
  else:
    order = [a_key, b_key]

  history: list[list[str]] = []
  transcript = []

  for i in range(turns):
    speaker_key = order[i % 2]
    listener_key = order[(i + 1) % 2]
    speaker = agents[speaker_key]
    listener = agents[listener_key]
    speaker_name = _agent_name(speaker_key, speaker, display_keys)
    listener_name = _agent_name(listener_key, listener, display_keys)

    relationship = _get_relationship(config, speaker_key, listener_key)
    opening_instruction = opening if i == 0 and opening else ""
    context = _build_context(
      topic=topic,
      speaker_key=speaker_key,
      listener_key=listener_key,
      speaker_name=speaker_name,
      listener_name=listener_name,
      relationship=relationship,
      opening=opening_instruction,
      max_words=max_words,
      strict=strict
    )

    utterance = speaker.utterance(history, context=context)
    if display_keys:
      # Strip any leading bracketed speaker labels the model invents.
      utterance = utterance.strip()
      if utterance.startswith("[") and "]" in utterance:
        utterance = utterance.split("]", 1)[1].lstrip()
    history.append([speaker_name, utterance])
    transcript.append({"speaker": speaker_name, "text": utterance})
    print(f"{speaker_name}: {utterance}")

  if out_path:
    out_payload = {
      "topic": topic,
      "agents": agent_keys,
      "turns": turns,
      "transcript": transcript
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2))


def main():
  parser = argparse.ArgumentParser(
    description="Run a two-agent conversation with a fixed topic and relationships."
  )
  parser.add_argument("--config", default="conversation_config.example.json")
  parser.add_argument("--topic", required=True)
  parser.add_argument("--agents", required=True,
                      help="Two agent keys separated by a comma, e.g. alice,bob")
  parser.add_argument("--turns", type=int, default=6)
  parser.add_argument("--start", default=None,
                      help="Agent key to start the conversation")
  parser.add_argument("--opening", default="Start with a brief greeting and introduce the topic.")
  parser.add_argument("--max-words", type=int, default=None)
  parser.add_argument("--display-keys", action="store_true",
                      help="Use config keys as display names instead of agent full names")
  parser.add_argument("--strict", action="store_true",
                      help="Enforce stricter topic/relationship instructions in the prompt")
  parser.add_argument("--out", default=None,
                      help="Optional path to save transcript JSON")

  args = parser.parse_args()
  agent_keys = [k.strip() for k in args.agents.split(",") if k.strip()]
  out_path = Path(args.out).expanduser().resolve() if args.out else None
  run_conversation(
    config_path=Path(args.config).expanduser().resolve(),
    topic=args.topic,
    agent_keys=agent_keys,
    turns=args.turns,
    start_key=args.start,
    opening=args.opening,
    max_words=args.max_words,
    out_path=out_path,
    display_keys=args.display_keys,
    strict=args.strict
  )


if __name__ == "__main__":
  main()
