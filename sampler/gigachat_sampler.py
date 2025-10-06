import os
import time
import json
from typing import Any

from .._types import MessageList, SamplerBase, SamplerResponse

try:
    from gigachat import GigaChat as GigaChatSyncClient
    from gigachat.models.chat import Chat
    from gigachat.models.messages import Messages
    from gigachat.models.messages_role import MessagesRole
except Exception as e:  # pragma: no cover
    raise ImportError(
        "GigaChat SDK is required. Please install 'gigachat' package."
    ) from e


class GigaChatSampler(SamplerBase):
    """
    Sampler that calls GigaChat chat completions via the official SDK.
    """

    def __init__(
        self,
        model: str | None = None,
        system_message: str | None = None,
        temperature: float | None = 0.5,
        max_tokens: int | None = 1024,
        scope: str | None = None,
        base_url: str | None = None,
        auth_url: str | None = None,
        verify_ssl_certs: bool | None = None,
        max_retries: int | None = None,
        backoff_base: int | None = None,
    ) -> None:
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Read environment with sensible precedence: explicit args override env
        env_scope = os.environ.get("GIGA_SCOPE")
        env_base_url = os.environ.get("GIGA_BASE_URL")
        env_auth_url = os.environ.get("GIGA_AUTH_URL")
        env_verify_ssl_certs = os.environ.get("GIGA_VERIFY_SSL_CERTS")

        self.scope = scope or env_scope
        self.base_url = base_url or env_base_url
        self.auth_url = auth_url or env_auth_url
        self.verify_ssl_certs = verify_ssl_certs if verify_ssl_certs is not None else env_verify_ssl_certs

        # Auth methods from env
        self.credentials = os.environ.get("GIGA_CREDENTIALS")
        self.user = os.environ.get("GIGA_USER")
        self.password = os.environ.get("GIGA_PASSWORD")
        self.access_token = os.environ.get("GIGA_ACCESS_TOKEN")

        # Require at least one auth method
        if not (self.credentials or self.access_token or (self.user and self.password)):
            raise RuntimeError(
                "GigaChat auth not configured. Provide GIGA_CREDENTIALS or GIGA_ACCESS_TOKEN or GIGA_USER/PASSWORD."
            )

        # Retry customization (env overrides defaults, explicit args override env)
        env_max_retries = os.environ.get("GIGA_MAX_RETRIES")
        env_backoff_base = os.environ.get("GIGA_BACKOFF_BASE")

        try:
            env_max_retries_int = int(env_max_retries) if env_max_retries is not None else None
        except Exception:
            env_max_retries_int = None
        try:
            env_backoff_base_int = int(env_backoff_base) if env_backoff_base is not None else None
        except Exception:
            env_backoff_base_int = None

        self.max_retries = max_retries if max_retries is not None else (env_max_retries_int if env_max_retries_int is not None else 5)
        self.backoff_base = backoff_base if backoff_base is not None else (env_backoff_base_int if env_backoff_base_int is not None else 2)

        # Build and keep a persistent SDK client to avoid repeated OAuth under concurrency
        client_kwargs: dict[str, Any] = {}
        if self.base_url is not None:
            client_kwargs["base_url"] = self.base_url
        if self.auth_url is not None:
            client_kwargs["auth_url"] = self.auth_url
        if self.scope is not None:
            client_kwargs["scope"] = self.scope
        if self.verify_ssl_certs is not None:
            client_kwargs["verify_ssl_certs"] = self.verify_ssl_certs

        # Prefer preissued access token to skip OAuth
        if self.access_token:
            client_kwargs["access_token"] = self.access_token
        elif self.credentials:
            client_kwargs["credentials"] = self.credentials
        elif self.user and self.password:
            client_kwargs["user"] = self.user
            client_kwargs["password"] = self.password

        self._client = GigaChatSyncClient(**client_kwargs)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()

    def _pack_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": role, "content": content}

    def _to_gigachat_messages(self, message_list: MessageList) -> list[Messages]:
        giga_chat_messages: list[Messages] = []
        for msg in message_list:
            role_raw = str(msg.get("role"))
            # Normalize roles to GigaChat-supported ones
            role_map = {
                "user": "user",
                "assistant": "assistant",
                "system": "system",
                "function": "function",
                # Common alternates from other providers
                "developer": "system",
                "tool": "function",
            }
            role = role_map.get(role_raw, "user")

            content_val = msg.get("content", "")
            if content_val is None:
                content_str = ""
            elif isinstance(content_val, str):
                content_str = content_val
            else:
                # Messages.content is a string in GigaChat; serialize complex content
                try:
                    content_str = json.dumps(content_val, ensure_ascii=False)
                except Exception:
                    content_str = str(content_val)

            giga_chat_messages.append(Messages(role=MessagesRole(role), content=content_str))
        return giga_chat_messages

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        # Prepend system message if provided
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list

        trial = 0

        def _retry_sleep(current_trial: int, kind: str, error: Exception) -> int:
            exception_backoff = (self.backoff_base ** current_trial) if self.backoff_base and self.backoff_base > 1 else (2 ** current_trial)
            print(f"GigaChat {kind} error; retry {current_trial} after {exception_backoff} sec", error)
            time.sleep(exception_backoff)
            return current_trial + 1

        while True:
            try:
                # Build Chat payload
                chat_payload = Chat(
                    model=self.model,
                    messages=self._to_gigachat_messages(message_list),
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                completion = self._client.chat(chat_payload)

                # Parse response
                content = completion.choices[0].message.content if completion.choices else ""
                usage = getattr(completion, "usage", None)
                x_headers = getattr(completion, "x_headers", None)
                metadata: dict[str, Any] = {
                    "usage": usage.dict() if hasattr(usage, "dict") and usage else None,
                    "x_headers": x_headers if x_headers else None,
                    "model": getattr(completion, "model", None),
                    "created": getattr(completion, "created", None),
                    "thread_id": getattr(completion, "thread_id", None),
                    "message_id": getattr(completion, "message_id", None),
                }
                return SamplerResponse(
                    response_text=content or "",
                    actual_queried_message_list=message_list,
                    response_metadata=metadata,
                )
            except Exception as e:
                # Retry on any error; SDK-specific exceptions may vary by version
                trial = _retry_sleep(trial, "generic", e)
                if trial >= self.max_retries:
                    raise


