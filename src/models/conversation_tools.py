from typing import Optional, Any
from ..utils.reusable_classes import (
    Message, UserMessage, AssistantMessage, SystemMessage, RoleType, Conversation
)


def align_message_roles(
    messages: list[Message],
    next_agent_id: str,
    real_user_agent_id: Optional[str] = None,
) -> list[Message]:

    if not messages:
        return []

    # Get the id of the other agent if it exists
    other_agent_ids = list(set(
        [message.agent_id for message in messages
         if message.agent_id is not None and message.agent_id not in [
             next_agent_id, real_user_agent_id]]
    ))
    other_agent_id = None
    if other_agent_ids:
        if len(other_agent_ids) > 1:
            raise ValueError(
                f"More than two agents found in the conversation:\n"
                f"- other_agent_ids: {other_agent_ids}\n"
                f"- next_agent_id: {next_agent_id}\n"
                f"- real_user_id: {real_user_agent_id}."
            )
        other_agent_id = other_agent_ids[0]

    # Next agent will act as the assistant, so make sure the type mapping is correct
    role_map: dict[str, type[Message]] = {next_agent_id: AssistantMessage}
    if other_agent_id:
        role_map[other_agent_id] = UserMessage

    aligned = []
    for i, message in enumerate(messages[::-1]):
        if real_user_agent_id and message.agent_id == real_user_agent_id:
            aligned.append(message)
            continue
        if message.role == RoleType.SYSTEM:
            aligned.append(message)
            continue

        if message.agent_id not in role_map:
            raise ValueError(
                f"Agent ID {message.agent_id} not found in role mapping. "
                "Ensure that the agent IDs are correctly set in the messages."
            )

        msg_type = role_map[message.agent_id]
        aligned.append(msg_type(
            content=message.content,
            content_type=message.content_type,
            model_name=message.model_name,
            model_provider=message.model_provider,
            agent_id=message.agent_id
        ))

    return aligned[::-1]


def annotate_messages_with_prefix(
    messages: list[Message],
    map_by_role: Optional[dict[RoleType, str | None]] = None,
    map_by_list: Optional[list[str]] = None,
    map_by_agent_id: Optional[dict[str, str | None]] = None,
    agent_id_overrides: Optional[dict[str, str | None]] = None,
) -> list[Message]:
    """Annotate message content with role-specific prefixes.

    Args:
        messages (list[Message]): List of messages to annotate.
        map_by_role (Optional[dict]): Dictionary mapping roles to annotations.
        map_by_list (Optional[list[str]]): List of annotations corresponding to each message.
        map_by_agent_id (Optional[dict[str, str]]): Dictionary mapping agent IDs to annotations.
        agent_id_overrides (Optional[dict[str, str]]): Dictionary to override agent IDs for annotation.
    Returns:
        list[Message]: List of annotated messages.
    Raises:
        ValueError: If both map_by_role and map_by_list are provided, or if their lengths do not match.
    """

    if not map_by_role and not map_by_list and not map_by_agent_id:
        return messages

    if (map_by_role and map_by_list) or (map_by_role and map_by_agent_id) or (map_by_list and map_by_agent_id):
        raise ValueError(
            "Only one mapping strategy can be used at a time")

    if isinstance(map_by_list, list) and len(map_by_list) != len(messages):
        raise ValueError(
            "message_mapping must be a list with the same length as messages.")

    annotated_messages = []
    for i, message in enumerate(messages):

        if message.role == RoleType.SYSTEM:
            # System messages are not annotated
            annotated_messages.append(message)
            continue

        if map_by_role:
            annotation = map_by_role.get(message.role, "")
        elif map_by_list:
            annotation = map_by_list[i]
        elif map_by_agent_id:
            if not message.agent_id:
                raise ValueError(
                    "message.agent_id must be set when using map_by_agent_id.")
            annotation = map_by_agent_id.get(message.agent_id)
        else:
            raise ValueError(
                "Either role_mapping or message_mapping must be provided.")

        if agent_id_overrides and message.agent_id in agent_id_overrides:
            annotation = agent_id_overrides[message.agent_id]

        if annotation:
            if message.role == RoleType.USER:
                msg = UserMessage
            elif message.role == RoleType.ASSISTANT:
                msg = AssistantMessage
            else:
                raise ValueError(
                    f"Unsupported message role: {message.role}")
            # Carry the identity and reasoning metadata across: callers assign
            # the result back over their message list, and align_message_roles
            # needs agent_id on every message.
            annotated_message = msg(
                content=f"[{annotation}]: {message.content}",
                content_type=message.content_type,
                agent_id=message.agent_id,
                model_name=message.model_name,
                model_provider=message.model_provider,
                reasoning_content=message.reasoning_content,
                reasoning_tokens=message.reasoning_tokens,
            )
            annotated_messages.append(annotated_message)
        else:
            # If no annotation, keep the original message
            annotated_messages.append(message)
    return annotated_messages


def conversation_to_dicts(
    result: Conversation,
    skip_system: bool = True,
    skip_agent_id: Optional[str] = None
) -> list[dict[str, Any]]:
    """
    Annotate messages with model names and providers.

    Args:
        result (Conversation): The conversation result containing messages.
        skip_system (bool): Whether to skip system messages.
        skip_agent_id (Optional[str]): Agent ID to skip in the annotation.

    Returns:
        list[dict[str, Any]]: List of annotated messages.
    """
    annotated_messages = []
    # Skip first two system and user messages
    for i, msg in enumerate(result.messages):

        if msg.role == RoleType.SYSTEM and skip_system:
            continue

        if skip_agent_id and msg.agent_id == skip_agent_id:
            continue

        annotated_msg = {
            "agent_id": msg.agent_id,
            "model_name": msg.model_name,
            "model_provider": msg.model_provider,
            "message": msg.content,
            "turn": i,
        }

        # Only emitted for reasoning-model turns, so that transcripts from
        # non-reasoning runs are unchanged.
        if msg.reasoning_tokens is not None:
            annotated_msg["reasoning_tokens"] = msg.reasoning_tokens
        if msg.reasoning_content:
            annotated_msg["reasoning_content"] = msg.reasoning_content

        annotated_messages.append(annotated_msg)

    return annotated_messages
