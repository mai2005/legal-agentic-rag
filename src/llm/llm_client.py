import os
import json
import re
from pathlib import Path
from typing import Any
import httpx

class LLMClientError(RuntimeError):
    pass

class LLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        prompts_dir: str | Path = "src/prompts",
        timeout: float = 60.0,
        max_retries: int = 3
    ) -> None:
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "ollama")
        self.model = model or os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
        
        self.prompts_dir = Path(prompts_dir)
        self.timeout = timeout
        self.max_retries = max_retries

        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key and self.api_key.lower() != "ollama":
            headers["Authorization"] = f"Bearer {self.api_key}"

        self.client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout)
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _load_prompt(self, filename: str, fallback: str) -> str:
        prompt_path = self.prompts_dir / filename
        if prompt_path.exists():
            try:
                content = prompt_path.read_text(encoding="utf-8").strip()
                if content:
                    return content
            except Exception:
                pass
        return fallback

    def _format_prompt(self, template: str, **kwargs: Any) -> str:
        result = template
        for key, value in kwargs.items():
            placeholder = "{" + key + "}"
            if placeholder not in result:
                continue
            if isinstance(value, (dict, list, tuple)):
                val_str = json.dumps(value, ensure_ascii=False, indent=2)
            else:
                val_str = "" if value is None else str(value)
            result = result.replace(placeholder, val_str)
        return result

    def _request(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.post("/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    import time
                    time.sleep(2 ** (attempt - 1))

        raise LLMClientError(f"Không thể kết nối với LLM tại {self.base_url} sau {self.max_retries} lần thử.") from last_error

    def _parse_json(self, text: str) -> Any:
        text = text.strip()
        
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
        else:
            json_str = text

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise LLMClientError(f"Không thể parse JSON từ phản hồi LLM: {text}") from e

    def analyze_query(self, query: str) -> dict[str, Any]:
        fallback_prompt = (
            "Bạn là một trợ lý phân tích câu hỏi pháp lý. Hãy phân loại câu hỏi của người dùng và trích xuất thực thể.\n"
            "Hãy trả về kết quả dưới định dạng JSON với cấu trúc chính xác như sau:\n"
            "{\n"
            "  \"query_type\": \"semantic\" hoặc \"legal_relation\" hoặc \"legal_lookup\",\n"
            "  \"entities\": [danh sách các thực thể như tên luật, chương, điều, tên tổ chức, cá nhân...],\n"
            "  \"metadata\": {}\n"
            "}"
        )
        system_prompt = self._load_prompt("analyze_query.txt", fallback_prompt)
        user_prompt = f"Câu hỏi cần phân tích:\n{query}"
        
        response = self._request(system_prompt, user_prompt, json_mode=True)
        return self._parse_json(response)

    def decompose_query(self, query: str, max_sub_queries: int = 4) -> list[dict[str, Any]]:
        fallback_prompt = (
            "Bạn là trợ lý pháp lý chuyên nghiệp.\n"
            "Hãy phân tích và chia nhỏ câu hỏi phức tạp của người dùng thành tối đa {max_sub_queries} câu hỏi con độc lập.\n"
            "Mỗi câu hỏi con phải tập trung vào một khía cạnh riêng biệt để dễ dàng tìm kiếm trong cơ sở dữ liệu pháp lý.\n"
            "Trả về kết quả dưới dạng danh sách JSON (JSON Array) gồm các object có cấu trúc như sau:\n"
            "[\n"
            "  {\n"
            "    \"id\": \"sub_q1\",\n"
            "    \"query\": \"nội dung câu hỏi con 1 bằng tiếng Việt...\",\n"
            "    \"purpose\": \"mục đích tìm kiếm của câu hỏi con này...\",\n"
            "    \"dependencies\": []\n"
            "  }\n"
            "]"
        )
        
        system_prompt = self._load_prompt("decompose_query.txt", fallback_prompt)
        system_prompt = self._format_prompt(system_prompt, max_sub_queries=max_sub_queries)
        
        user_prompt = f"Câu hỏi gốc:\n{query}"
        
        response = self._request(system_prompt, user_prompt, json_mode=True)
        data = self._parse_json(response)
        
        if isinstance(data, dict) and "sub_queries" in data:
            data = data["sub_queries"]
        if not isinstance(data, list):
            raise LLMClientError("Decompose query phải trả về một danh sách JSON")
        return data

    def plan_dependencies(self, sub_queries: list[dict[str, Any]]) -> dict[str, list[str]]:
        fallback_prompt = (
            "Bạn là một chuyên gia quản lý tiến trình.\n"
            "Dựa trên danh sách các câu hỏi con pháp lý được cung cấp, hãy lập bản đồ phụ thuộc.\n"
            "Xác định câu hỏi nào cần có thông tin/câu trả lời từ câu hỏi khác trước khi thực hiện.\n"
            "Hãy trả về kết quả định dạng JSON biểu diễn các mối quan hệ phụ thuộc như sau:\n"
            "{\n"
            "  \"sub_q2\": [\"sub_q1\"],\n"
            "  \"sub_q1\": []\n"
            "}"
        )
        system_prompt = self._load_prompt("dependency_planner.txt", fallback_prompt)
        user_prompt = f"Danh sách câu hỏi con:\n{json.dumps(sub_queries, ensure_ascii=False, indent=2)}"
        
        response = self._request(system_prompt, user_prompt, json_mode=True)
        data = self._parse_json(response)
        if isinstance(data, dict):
            return data
        return {}

    def rewrite_query(
        self, 
        query: str, 
        query_type: str | None = None,
        entities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        dependency_context: str | None = None
    ) -> str:
        fallback_prompt = (
            "Bạn là trợ lý viết lại truy vấn tìm kiếm.\n"
            "Nhiệm vụ của bạn là viết lại câu hỏi pháp lý của người dùng thành một câu truy vấn tìm kiếm RAG ngắn gọn, "
            "tập trung vào các từ khóa luật pháp quan trọng, loại bỏ các từ cảm thán và cấu trúc ngữ pháp dư thừa."
        )
        system_prompt = self._load_prompt("rewrite_query.txt", fallback_prompt)
        system_prompt = self._format_prompt(
            system_prompt,
            query_type=query_type or "không rõ",
            query=query,
            entities=entities or [],
            metadata=metadata or {},
            dependency_context=dependency_context or "không có"
        )
        
        user_prompt = "Hãy thực hiện viết lại câu truy vấn ban đầu."
        
        response = self._request(system_prompt, user_prompt, json_mode=True)
        data = self._parse_json(response)
        if isinstance(data, dict) and "rewritten_query" in data:
            return str(data["rewritten_query"]).strip()
        return response.strip()

    def generate_answer(
        self, 
        query: str, 
        evidence: list[dict[str, Any]],
        query_type: str | None = None,
        metadata: dict[str, Any] | None = None
    ) -> str:
        fallback_prompt = (
            "Bạn là một luật sư giàu kinh nghiệm tư vấn pháp luật.\n"
            "Nhiệm vụ của bạn là trả lời câu hỏi của người dùng dựa TRÊN VÀ CHỈ TRÊN các bằng chứng pháp lý (văn bản luật) được cung cấp dưới đây.\n"
            "Yêu cầu:\n"
            "1. Chỉ sử dụng thông tin có trong phần bằng chứng.\n"
            "2. Trích dẫn chính xác số Điều, Khoản, tên Luật (ví dụ: 'Theo Điều 5 Luật Doanh nghiệp 2020...').\n"
            "3. Nếu bằng chứng không chứa câu trả lời, hãy trả lời trung thực là 'Không tìm thấy căn cứ pháp lý phù hợp trong dữ liệu cung cấp', không được tự bịa ra thông tin."
        )
        system_prompt = self._load_prompt("final_answer.txt", fallback_prompt)
        
        formatted_evidence = []
        for idx, item in enumerate(evidence, 1):
            doc_title = item.get("document_title", "Không rõ nguồn")
            article = item.get("article_number", "")
            content = item.get("content", "")
            formatted_evidence.append(f"[{idx}] Văn bản: {doc_title} - Điều {article}\nNội dung: {content}")
        evidence_str = "\n\n".join(formatted_evidence)

        system_prompt = self._format_prompt(
            system_prompt,
            query=query,
            evidence=evidence_str,
            query_type=query_type or "không rõ",
            metadata=metadata or {}
        )
        
        user_prompt = "Hãy tổng hợp và tạo câu trả lời pháp lý hoàn chỉnh dựa trên các bằng chứng được cung cấp."
        
        return self._request(system_prompt, user_prompt, json_mode=False)

    def verify_answer(self, query: str, answer: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
        fallback_prompt = (
            "Bạn là một chuyên gia thẩm định và kiểm soát chất lượng câu trả lời pháp lý.\n"
            "Hãy đối chiếu câu trả lời được cung cấp với danh sách bằng chứng gốc.\n"
            "Kiểm tra xem câu trả lời có chứa thông tin nào sai lệch, xuyên tạc hoặc không có căn cứ trong bằng chứng gốc hay không (Hallucination).\n"
            "Hãy trả về kết quả dưới định dạng JSON với cấu trúc chính xác:\n"
            "{\n"
            "  \"passed\": true (nếu câu trả lời trung thực và chính xác với bằng chứng) hoặc false (nếu có lỗi sai hoặc thông tin bịa đặt),\n"
            "  \"issues\": [danh sách các lỗi phát hiện được, nếu passed=true thì để danh sách rỗng]\n"
            "}"
        )
        system_prompt = self._load_prompt("verify_answer.txt", fallback_prompt)
        
        formatted_evidence = []
        for idx, item in enumerate(evidence, 1):
            doc_title = item.get("document_title", "Không rõ nguồn")
            article = item.get("article_number", "")
            content = item.get("content", "")
            formatted_evidence.append(f"[{idx}] Văn bản: {doc_title} - Điều {article}\nNội dung: {content}")
        evidence_str = "\n\n".join(formatted_evidence)

        system_prompt = self._format_prompt(
            system_prompt,
            query=query,
            draft_answer=answer,
            evidence=evidence_str
        )
        
        user_prompt = "Hãy thực hiện thẩm định và trả về kết quả kiểm định dưới dạng JSON."
        
        response = self._request(system_prompt, user_prompt, json_mode=True)
        return self._parse_json(response)
