"""Локальный клиент LM Studio (OpenAI-совместимый API), только через loopback."""

from __future__ import annotations

import http.client
import json
import os
import re
from typing import Any

LOOPBACK_HOST = "127.0.0.1"
DEFAULT_PORT = 1234
JSON_ONLY_HINT = (
	"\n\nОтвечай ТОЛЬКО одним JSON-объектом, без пояснений, без markdown и без текста вокруг."
)


class LocalLLMError(RuntimeError):
	"""Ошибка обращения к локальной модели (с понятным текстом для пользователя)."""


def _extract_json(text: str) -> dict[str, Any]:
	"""Достаёт JSON-объект из ответа, даже если модель добавила лишний текст."""
	candidates = [text.strip()]
	fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
	if fenced:
		candidates.append(fenced.group(1).strip())
	start, end = text.find("{"), text.rfind("}")
	if start != -1 and end > start:
		candidates.append(text[start : end + 1])
	for candidate in candidates:
		try:
			parsed = json.loads(candidate)
		except json.JSONDecodeError:
			continue
		if isinstance(parsed, dict):
			return parsed
	raise LocalLLMError(f"Модель вернула не JSON-объект: {text[:300]!r}")


class LLMBackend:
	"""Запрашивает JSON-ответ у модели, запущенной в LM Studio на этом компьютере."""

	def __init__(
		self,
		model: str | None = None,
		*,
		port: int | None = None,
		timeout: float = 300.0,
		retries: int = 2,
	) -> None:
		"""Имя модели берётся из аргумента или LLM_MODEL; если не задано, из списка LM Studio."""
		self.model = model or os.getenv("LLM_MODEL") or os.getenv("LMSTUDIO_MODEL")
		self.port = int(port or os.getenv("LMSTUDIO_PORT", DEFAULT_PORT))
		self.timeout = timeout
		self.retries = retries

	# --- низкоуровневый HTTP -------------------------------------------------

	def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
		"""Выполняет запрос к локальному серверу и возвращает разобранный JSON."""
		payload = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
		headers = {"Content-Type": "application/json"} if payload is not None else {}
		connection = http.client.HTTPConnection(LOOPBACK_HOST, self.port, timeout=self.timeout)
		try:
			connection.request(method, path, body=payload, headers=headers)
			response = connection.getresponse()
			raw = response.read().decode("utf-8", errors="replace")
		except OSError as exc:
			raise LocalLLMError(
				f"Не удалось подключиться к LM Studio на {LOOPBACK_HOST}:{self.port}. "
				"Запустите сервер (вкладка Developer -> Start Server или `lms server start`) "
				"и загрузите модель."
			) from exc
		finally:
			connection.close()

		if response.status != 200:
			raise LocalLLMError(f"LM Studio вернула HTTP {response.status}: {raw[:500]}")
		try:
			return json.loads(raw)
		except json.JSONDecodeError as exc:
			raise LocalLLMError(f"LM Studio вернула не JSON: {raw[:300]!r}") from exc

	# --- модель --------------------------------------------------------------

	def available_models(self) -> list[str]:
		"""Возвращает идентификаторы моделей, которые видит сервер."""
		data = self._request("GET", "/v1/models").get("data", [])
		return [item["id"] for item in data if isinstance(item, dict) and "id" in item]

	def resolve_model(self) -> str:
		"""Определяет имя модели: заданное пользователем или первая подходящая из списка."""
		if self.model:
			return self.model
		models = [m for m in self.available_models() if "embed" not in m.lower()]
		if not models:
			raise LocalLLMError("В LM Studio нет доступных моделей. Загрузите ru-uncensored-v2.")
		self.model = models[0]
		return self.model

	def preflight(self) -> None:
		"""Проверяет, что сервер отвечает и нужная модель есть в списке."""
		models = self.available_models()
		if self.model and models and self.model not in models:
			raise LocalLLMError(
				f"Модель {self.model!r} не найдена в LM Studio. Доступные: {', '.join(models)}. "
				"Задайте точное имя через LLM_MODEL."
			)
		self.resolve_model()


	def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
		"""Отправляет сообщения модели и возвращает JSON-объект из её ответа."""
		body = {
			"model": self.resolve_model(),
			"messages": [
				{"role": "system", "content": system_prompt + JSON_ONLY_HINT},
				{"role": "user", "content": user_prompt},
			],
			"temperature": 0.2,
			"max_tokens": 4096,
			"stream": False,
		}
		last_error: Exception | None = None
		for _ in range(self.retries + 1):
			result = self._request("POST", "/v1/chat/completions", body)
			if "error" in result:
				raise LocalLLMError(f"LM Studio сообщила об ошибке: {result['error']}")
			choices = result.get("choices")
			if not choices:
				raise LocalLLMError(f"Неожиданный ответ LM Studio: {str(result)[:300]}")
			content = (choices[0].get("message") or {}).get("content") or ""
			try:
				return _extract_json(content)
			except LocalLLMError as exc:
				last_error = exc
		assert last_error is not None
		raise last_error
