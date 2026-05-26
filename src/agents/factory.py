from __future__ import annotations

import json
import shutil
from pathlib import Path

import autogen
from autogen import register_function

from config import SAGEConfig
from prompts import PLANNER_PROMPT, ENGINEER_PROMPT, CRITIC_PROMPT, SUMMARIZER_PROMPT
from knowledge.tools import graph_source_rag
from events import emit_next_speaker, emit_visible_groupchat_message


def build_agents(config: SAGEConfig):
    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        system_message="Human admin interacting with the agents.",
        human_input_mode="NEVER",
        code_execution_config={"use_docker": False},
    )

    planner = autogen.AssistantAgent(
        name="planner",
        llm_config=config.llm_config,
        system_message=PLANNER_PROMPT,
    )

    engineer = autogen.AssistantAgent(
        name="engineer",
        llm_config=config.llm_tool_config,
        system_message=ENGINEER_PROMPT,
    )

    critic = autogen.AssistantAgent(
        name="critic",
        llm_config=config.llm_config,
        system_message=CRITIC_PROMPT,
    )

    summarizer = autogen.AssistantAgent(
        name="summarizer",
        llm_config=config.llm_config_summarizer,
        system_message=SUMMARIZER_PROMPT,
    )

    desc_graph_src_rag = (
        "Graph Source RAG: Return KG paths and their source chunks related to the query. "
        "When there is a main query argument, put the unchanged full query there to retrieve the full content."
    )

    register_function(
        graph_source_rag,
        caller=engineer,
        executor=user_proxy,
        description=desc_graph_src_rag,
    )

    return user_proxy, planner, engineer, critic, summarizer


def is_nonempty_text(x):
    return isinstance(x, str) and bool(x.strip())


def extract_text(msg):
    if msg is None:
        return ""
    if isinstance(msg, str):
        return msg.strip()
    if isinstance(msg, dict):
        content = msg.get("content", "")
        if isinstance(content, str):
            return content.strip()
        return str(content).strip()
    return str(msg).strip()


def last_message(groupchat: autogen.GroupChat):
    return groupchat.messages[-1] if groupchat.messages else {}


def has_tool_call(msg):
    return isinstance(msg, dict) and bool(msg.get("tool_calls") or msg.get("function_call"))


def is_question_message(msg):
    return "QUESTION:" in extract_text(msg)


def is_write_report_message(msg):
    text = extract_text(msg).upper()
    return "WRITE REPORT" in text or "WRITE_REPORT" in text


def is_terminate_message(msg):
    return "TERMINATE" in extract_text(msg).upper()


def prune_messages(groupchat: autogen.GroupChat, agent_name: str, start: int = 0, remove_all: bool = False, role_filter: str | None = None, last_move_to_last: bool = False):
    msg_idxs = []
    for i, m in enumerate(groupchat.messages):
        if not isinstance(m, dict):
            continue
        if m.get("name") != agent_name:
            continue
        if role_filter is not None and m.get("role") != role_filter:
            continue
        msg_idxs.append(i)

    if len(msg_idxs) <= 2 and not remove_all:
        return

    remove = msg_idxs[start:-1]
    if remove_all:
        remove = msg_idxs
        return

    if last_move_to_last:
        groupchat.messages.append(groupchat.messages[msg_idxs[-1]])
        groupchat.messages.pop(msg_idxs[-1])

    for i in reversed(remove):
        groupchat.messages.pop(i)


def make_speaker_selection(user_proxy, planner, engineer, critic, summarizer):
    # Notebook-faithful speaker function. No extra guards/state machine.
    def speaker_selection_func(last_speaker, groupchat: autogen.GroupChat):
        # Stream the visible message that was just appended by AutoGen.
        # This is structured event output, not hidden chain-of-thought.
        emit_visible_groupchat_message(groupchat)

        next_speaker = planner

        if len(groupchat.messages) <= 1:
            next_speaker = planner

        last_msg = last_message(groupchat)

        if is_terminate_message(last_msg):
            next_speaker = None

        if last_speaker is user_proxy:
            prev = groupchat.messages[-1]
            if prev.get("role") == "tool":
                next_speaker = engineer
            else:
                next_speaker = planner

        if last_speaker is planner:
            if is_write_report_message(last_msg):
                next_speaker = summarizer
            elif is_question_message(last_msg):
                next_speaker = engineer
            else:
                next_speaker = planner

        if last_speaker is engineer:
            next_speaker = user_proxy if has_tool_call(last_msg) else critic

        if last_speaker is critic:
            next_speaker = planner

        if last_speaker is summarizer:
            next_speaker = None

        emit_next_speaker(next_speaker)
        prune_messages(groupchat, "user_proxy", role_filter="tool")
        return next_speaker

    return speaker_selection_func


def build_groupchat(user_proxy, planner, engineer, critic, summarizer):
    agents = [user_proxy, planner, engineer, critic, summarizer]
    groupchat = autogen.GroupChat(
        agents=agents,
        messages=[],
        max_round=100,
        speaker_selection_method=make_speaker_selection(user_proxy, planner, engineer, critic, summarizer),
        select_speaker_auto_llm_config=None,
    )
    manager = autogen.GroupChatManager(groupchat)
    return agents, groupchat, manager


def reset_all_agents(agents, manager):
    for agent in agents:
        agent.reset()
    try:
        manager.reset()
    except AttributeError:
        pass


def clear_cache_dir() -> None:
    try:
        shutil.rmtree(".cache")
    except Exception:
        pass


def save_answer_outputs(output_dir: Path, idx_or_name: str, question: str, answer) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"qa_log_{idx_or_name}.txt", "a", encoding="utf-8") as f:
        f.write("Q:\n")
        f.write(str(question) + "\n\n")
        f.write("A:\n")
        f.write(str(answer) + "\n")
        f.write("\n" + "-" * 80 + "\n\n")
    chat_history = getattr(answer, "chat_history", None)
    if chat_history is not None:
        with open(output_dir / f"chat_result_{idx_or_name}.json", "w", encoding="utf-8") as f:
            json.dump(chat_history, f, indent=4, ensure_ascii=False, default=str)
