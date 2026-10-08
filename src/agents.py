"""
Agent 定义与 Prompt 模板

这个文件负责：
1. 定义 VisualAgent：通过 OpenAI-compatible API（例如 OpenRouter）分析 K 线图。
2. 定义 Quant Agent Prompt：生成 QuantReport。
3. 定义 News Agent Prompt：生成 NewsReport。
4. 定义 CIO Agent Prompt：生成 FinalDecision。

注意：
真正的 LangGraph 节点在 graph.py 里。
这里主要放“角色能力”和“提示词”。
"""

from __future__ import annotations

import base64
import ast
import json
import os
import re
import random
import time
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=True)

from src.models import VisualReport
from src.settings import output_language_instruction


# ============================================================
# 1. 工具函数：图片转 Base64
# ============================================================
def encode_image_to_base64(image_path: str) -> str:
    """
    将本地图片转成 base64 字符串。
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def guess_image_mime_type(image_path: str) -> str:
    """
    根据图片后缀猜测 MIME 类型。
    """
    suffix = Path(image_path).suffix.lower()

    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"

    if suffix == ".webp":
        return "image/webp"

    return "image/png"


def extract_json_from_text(text: str) -> dict[str, Any]:
    """
    从模型返回文本中提取 JSON。
    """
    content = text.strip()

    if content.startswith("```json"):
        content = content.removeprefix("```json").strip()
        if content.endswith("```"):
            content = content.removesuffix("```").strip()

    elif content.startswith("```"):
        content = content.removeprefix("```").strip()
        if content.endswith("```"):
            content = content.removesuffix("```").strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    start = content.find("{")
    end = content.rfind("}")

    if start != -1 and end != -1 and end > start:
        json_text = content[start: end + 1]
        return json.loads(json_text)

    raise json.JSONDecodeError(
        "No valid JSON object found in model response",
        content,
        0,
    )


# ============================================================
# 2. Visual Agent
# ============================================================
class VisualAgent:
    """$0 OpenRouter Vision Agent: VLM sees chart; local code validates JSON."""

    DEFAULT_VISION_POOL = (
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free,"
        "openrouter/free"
    )

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.base_url = os.getenv(
            "OPENAI_BASE_URL", "https://openrouter.ai/api/v1"
        ).strip()
        raw_models = os.getenv("VISION_MODELS", self.DEFAULT_VISION_POOL)
        blocked = {
            "nvidia/nemotron-3-nano-omni:free",
            "thinkingmachines/inkling-small:free",
        }
        self.models = list(dict.fromkeys(
            item.strip() for item in raw_models.split(",")
            if item.strip() and item.strip() not in blocked
        ))
        self.timeout = float(os.getenv("VISION_TIMEOUT_SECONDS", "90"))
        self.max_tokens = int(os.getenv("VISION_MAX_TOKENS", "1600"))
        self.max_retries = int(os.getenv("VISION_MAX_RETRIES", "1"))
        self.retry_base = float(os.getenv("VISION_RETRY_BASE_SECONDS", "4"))
        self.retry_max = float(os.getenv("VISION_RETRY_MAX_SECONDS", "12"))
        self.client = (
            OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
            if self.api_key else None
        )

    @staticmethod
    def _message_text(message: Any) -> str:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    value = item.get("text") or item.get("content")
                    if value:
                        parts.append(str(value))
                else:
                    value = getattr(item, "text", None)
                    if value:
                        parts.append(str(value))
            return "\n".join(parts).strip()
        return ""

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        if getattr(exc, "status_code", None) == 429:
            return True
        response = getattr(exc, "response", None)
        if getattr(response, "status_code", None) == 429:
            return True
        text = str(exc).lower()
        return "429" in text or "rate limit" in text or "rate-limited" in text

    @staticmethod
    def _is_non_retryable(exc: Exception) -> bool:
        msg = str(exc).lower()
        return (getattr(exc, "status_code", None) == 401
                or "missing authentication header" in msg
                or "free-models-per-day" in msg
                or "openrouter_free_tier_daily" in msg)

    def _delay(self, attempt: int) -> float:
        base = min(self.retry_max, self.retry_base * (2 ** attempt))
        return base + random.uniform(0.0, min(1.5, base * 0.25))

    def _create_with_retry(self, kwargs: dict, label: str):
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                if not self._is_non_retryable(exc) and self._is_rate_limit_error(exc) and attempt < self.max_retries:
                    delay = self._delay(attempt)
                    print(f"[RateLimit] {label}: waiting {delay:.1f}s before retry {attempt + 1}/{self.max_retries}...")
                    time.sleep(delay)
                    continue
                raise

    @staticmethod
    def _parse_visual_json(content: str) -> dict[str, Any]:
        """Zero-cost repair for common VLM JSON formatting imperfections."""
        try:
            return extract_json_from_text(content)
        except Exception as first_error:
            text = content.strip()
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end <= start:
                raise first_error
            candidate = text[start:end + 1]
            candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                return json.loads(candidate)
            except Exception:
                pass
            try:
                value = ast.literal_eval(candidate)
                if isinstance(value, dict):
                    return value
            except Exception:
                pass
            raise first_error

    def _request_model(self, model_name: str, prompt: str, image_url: str):
        # Deliberately DO NOT send response_format / require_parameters here.
        # Free multimodal endpoints may support image input without native JSON Schema.
        kwargs = {
            "model": model_name,
            "temperature": 0.1,
            "max_tokens": self.max_tokens,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }],
        }
        return self._create_with_retry(kwargs, f"Vision/{model_name}")

    def analyze_chart(self, image_path: str) -> VisualReport | str:
        if not image_path:
            return "No Chart Image Available"
        if not os.path.exists(image_path):
            return f"Chart image does not exist: {image_path}"
        if self.client is None:
            return "Visual Analysis Failed: OPENAI_API_KEY is missing."

        try:
            image_base64 = encode_image_to_base64(image_path)
            mime_type = guess_image_mime_type(image_path)
            image_url = f"data:{mime_type};base64,{image_base64}"
            prompt = f"""
你是一名 A 股图表研究 Agent。请真正查看所附 K 线图片，只根据图片中可见的价格走势、K线形态、均线、成交量和 MACD 做描述性技术分析。
不要编造图片中无法确认的数据，也不要输出 BUY/SELL 指令。

只返回一个 JSON object。不要 Markdown，不要代码围栏，不要在 JSON 前后添加解释：
{{
  "key_patterns": ["形态1"],
  "macd_signal": "neutral",
  "trend_direction": "sideways",
  "support_resistance": {{"support": 0.0, "resistance": 0.0}},
  "visual_conclusion": "视觉技术面结论"
}}

约束：
- key_patterns 是字符串数组。
- macd_signal 只能是 bullish_divergence, bearish_divergence, golden_cross, death_cross, neutral。
- trend_direction 只能是 strong_uptrend, weak_uptrend, sideways, weak_downtrend, strong_downtrend。
- support/resistance 必须是数字；无法可靠读出时用 0.0，禁止猜测。
{output_language_instruction()}
""".strip()

            failures: list[str] = []
            for index, model_name in enumerate(self.models, start=1):
                print(f"[VisionLLM] Vision: trying {model_name} ({index}/{len(self.models)})")
                try:
                    response = self._request_model(model_name, prompt, image_url)
                    if not getattr(response, "choices", None):
                        raise ValueError("vision model returned no choices")
                    choice = response.choices[0]
                    content = self._message_text(choice.message)
                    if not content:
                        raise ValueError(f"empty final content; finish_reason={getattr(choice, 'finish_reason', None)}")
                    data = self._parse_visual_json(content)
                    report = VisualReport.model_validate(data)
                    actual_model = getattr(response, "model", None) or model_name
                    print(f"[VisionLLM] Vision: success via {actual_model}")
                    return report
                except Exception as exc:
                    if self._is_non_retryable(exc):
                        return f"Visual Analysis Failed: non-retryable OpenRouter error: {str(exc)[:260]}"
                    preview = str(exc).replace("\n", " ")[:260]
                    failures.append(f"{model_name}: {preview}")
                    print(f"[VisionLLM] Vision: {model_name} failed ({preview}); switching model...")

            return "Visual Analysis Failed: all free vision models failed. " + " | ".join(failures[-2:])
        except Exception as exc:
            return f"Visual Analysis Failed: {type(exc).__name__}: {exc}"


# ============================================================
# 3. Quant Agent Prompt
# ============================================================
QUANT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
你是一名顶级 A 股量化研究员，专注于“资金驱动 + 趋势跟踪”模型。

你的核心任务不是预测股价一定涨跌，而是判断：
1. 主力资金更像是在吸筹、洗盘，还是出货。
2. 当前量价关系是否健康。
3. 技术指标是否支持继续持有或买入。
4. 当前风险等级是多少。

你必须严格按照下面 JSON 格式输出，不要输出 Markdown，不要输出额外解释：

{{
  "main_force_intent": "accumulating",
  "confidence_score": 0.75,
  "key_technical_signals": ["MACD金叉", "RSI偏强", "放量突破"],
  "price_volume_analysis": "缩量上涨，主力控盘明显",
  "risk_assessment": "medium",
  "detailed_analysis": "不少于100字的详细量化分析"
}}

字段要求：
- main_force_intent 只能是 accumulating, washing, distributing 三选一。
  - accumulating：吸筹，资金逐步进入。
  - washing：洗盘，震荡整理但未明显出货。
  - distributing：出货，放量滞涨或资金明显流出。
- confidence_score 必须是 0 到 1 之间的数字。
- key_technical_signals 必须是字符串列表，1 到 5 条。
- risk_assessment 只能是 low, medium, high 三选一。
- detailed_analysis 必须结合输入数据，不要空泛。
- 如果资金流数据出现 fetch failed，必须在 detailed_analysis 中明确说明数据不足。
""".strip(),
        ),
        (
            "human",
            """
以下是该股票的量化输入数据：

【Prophet 趋势预测】
{prophet_forecast}

【最近行情快照】
{market_data_str}

【资金流向数据】
{fund_flow_data}

请完成以下任务：
1. 优先分析资金流向，判断主力意图，权重约 60%。
2. 结合最近行情快照，判断量价配合情况。
3. 结合 Prophet 趋势预测，判断短期趋势是偏强、震荡还是偏弱。
4. 输出严格 JSON，字段必须和系统提示中的格式一致。

【输出语言要求】
{output_language_instruction}
""".strip(),
        ),
    ]
)


# ============================================================
# 4. News Agent Prompt
# ============================================================
NEWS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
你是一名资深 A 股新闻舆情分析师，也负责建立“标的档案”。

你的任务：
1. 根据输入新闻，提取公司基本信息。
2. 判断近期新闻情绪。
3. 过滤过期信息。
4. 区分“公司级新闻”和“行业级新闻”。
5. 找出潜在利好、利空和风险。
6. 给 CIO 提供可靠的消息面摘要。

你必须严格按照下面 JSON 格式输出，不要输出 Markdown，不要输出额外解释：

{{
  "company_profile": "公司名称、行业、板块、核心概念等信息，必须是字符串",
  "sentiment_score": 0.5,
  "key_news_summary": ["近期新闻摘要1", "近期新闻摘要2"],
  "market_positioning": "市场地位、行业位置、政策支持或竞争优势",
  "risk_warnings": ["风险1", "风险2"]
}}

关键判断规则：
- company_profile 必须是字符串，不是字典。
- sentiment_score 必须是 -1 到 1 之间的数字。
  - -1 表示极度负面。
  - 0 表示中性。
  - 1 表示极度正面。
- key_news_summary 必须是字符串列表，最多 5 条。
- risk_warnings 必须是字符串列表，最多 3 条。

新闻可靠性规则：
1. 只有同时满足“近期 + 直接相关 + 内容明确”的新闻，才可以作为强证据。
2. Published Date: Unknown 的新闻，不能当作强近期证据，只能作为弱证据或背景信息。
3. Relevance 是 possibly_industry_or_market_related 的新闻，只能作为行业背景，不能写成公司已经发生的事实。
4. 不要把行业新闻误写成公司公告。
5. 不要把旧新闻当作最新利好或利空。
6. 不要编造新闻中没有出现的财务数字、监管处罚、减持、增持、订单、政策影响。
7. 如果新闻数据不足、日期缺失、搜索失败，必须明确写进 risk_warnings。
8. 如果搜索结果不能直接支持某个结论，就不要把它写成确定事实。
""".strip(),
        ),
        (
            "human",
            """
以下是搜索到的新闻和舆情数据：

{news_data}

请完成以下任务：
1. 提取公司名称、行业、板块、核心概念，合并成 company_profile 字符串。
2. 过滤明显过期、无日期、无关或弱相关的信息。
3. 区分公司级新闻、行业级新闻、市场泛新闻。
4. 对近期消息面进行 sentiment_score 打分。
5. 总结 key_news_summary：
   - 每条摘要尽量说明它是“公司级证据”还是“行业级背景”。
   - 如果日期未知，必须写明“日期未知，证据较弱”。
6. 给出 market_positioning 和 risk_warnings。
7. 如果 Tavily 搜索失败、新闻不足、日期缺失或相关性弱，必须降低 sentiment_score 的绝对值，并写进 risk_warnings。
8. 输出严格 JSON，字段必须和系统提示中的格式一致。

【输出语言要求】
{output_language_instruction}
""".strip(),
        ),
    ]
)

# ============================================================
# 5. Bull / Bear / Risk Debate Prompts
# ============================================================
BULL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
你是牛方研究员，职责是构建“买入、加仓或继续持有”的最强论证。

你的目标不是盲目看多，而是在已有证据中寻找支持进攻或持有的合理依据。
你必须承认证据缺口，不能把弱证据包装成强证据。

你必须严格按照下面 JSON 格式输出，不要输出 Markdown，不要输出额外解释：

{{
  "stance": "HOLD",
  "strongest_points": ["支持理由1", "支持理由2"],
  "evidence_quality": "medium",
  "counter_to_bear_risks": "对潜在风险的回应",
  "invalidation_condition": "牛方观点失效条件"
}}

字段要求：
- stance 只能是 BUY 或 HOLD。牛方不能输出 SELL。
- strongest_points 必须是 1 到 5 条，且必须来自输入报告，不得编造数据。
- evidence_quality 只能是 strong, medium, weak。
- 如果资金流、新闻、视觉或行情数据缺失，evidence_quality 不能是 strong。
- counter_to_bear_risks 必须主动回应可能的破位、出货、利空或估值风险。
- invalidation_condition 必须是具体条件，不要写“市场不好”这种空话。
""".strip(),
        ),
        (
            "human",
            """
当前价格：{last_price} CNY

【标的档案】
{profile_data}

【视觉技术面报告】
{visual_report}

【量化资金面报告】
{quant_report}

【新闻舆情报告】
{news_report}

请站在牛方角度完成报告：
1. 找出支持继续持有、买入或加仓的最强证据。
2. 主动承认证据不足之处。
3. 预先回应熊方可能提出的风险。
4. 给出牛方观点失效条件。
5. 输出严格 JSON。

【输出语言要求】
{output_language_instruction}
""".strip(),
        ),
    ]
)


BEAR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
你是熊方研究员，职责是构建“卖出、减仓或谨慎持有”的最强论证，并反驳牛方。

你的目标不是盲目看空，而是从风险控制角度识别牛方论证中的弱点。
你必须针对牛方报告逐点反驳，不能只重复自己的观点。

你必须严格按照下面 JSON 格式输出，不要输出 Markdown，不要输出额外解释：

{{
  "stance": "SELL",
  "strongest_points": ["风险理由1", "风险理由2"],
  "evidence_quality": "medium",
  "rebuttal_to_bull_case": "针对牛方报告的反驳",
  "risk_trigger": "触发减仓或卖出的关键条件"
}}

字段要求：
- stance 只能是 SELL 或 HOLD。熊方不能输出 BUY。
- strongest_points 必须是 1 到 5 条，且必须来自输入报告，不得编造数据。
- evidence_quality 只能是 strong, medium, weak。
- rebuttal_to_bull_case 必须引用牛方报告里的具体观点，并指出其证据弱点或遗漏风险。
- 如果看空证据不足，stance 应为 HOLD，但仍需说明需要观察的风险触发条件。
- risk_trigger 必须具体，例如跌破某类支撑、主力连续流出、放量滞涨、明确公司级利空等。
""".strip(),
        ),
        (
            "human",
            """
当前价格：{last_price} CNY

【标的档案】
{profile_data}

【视觉技术面报告】
{visual_report}

【量化资金面报告】
{quant_report}

【新闻舆情报告】
{news_report}

【牛方报告】
{bull_report}

请站在熊方角度完成报告：
1. 找出支持卖出、减仓或谨慎持有的最强风险证据。
2. 针对牛方 strongest_points 和 counter_to_bear_risks 做具体反驳。
3. 判断风险证据质量。
4. 给出触发减仓或卖出的关键条件。
5. 输出严格 JSON。

【输出语言要求】
{output_language_instruction}
""".strip(),
        ),
    ]
)


RISK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
你是风控负责人，职责是裁决牛方和熊方辩论，并给 CIO 一个保守、可执行的风险建议。

你的核心原则：
1. 风险控制优先。
2. 不让单一 Agent 的强语气替代证据质量。
3. 当牛熊分歧较大或关键数据缺失时，必须限制最终置信度。
4. preferred_action 必须和仓位风险一致。

你必须严格按照下面 JSON 格式输出，不要输出 Markdown，不要输出额外解释：

{{
  "preferred_action": "HOLD",
  "confidence_cap": 0.65,
  "winning_side": "balanced",
  "key_risk_controls": ["风控措施1", "风控措施2"],
  "debate_summary": "牛熊分歧、证据质量和风控裁决总结"
}}

字段要求：
- preferred_action 只能是 BUY, SELL, HOLD。
- confidence_cap 必须是 0 到 1 之间的数字。
- winning_side 只能是 bull, bear, balanced。
- key_risk_controls 必须是 1 到 5 条具体措施。
- 如果任一关键报告显示数据缺失、fetch failed 或分析失败，confidence_cap 最高 0.65。
- 如果两个以上关键数据源缺失或失败，confidence_cap 最高 0.50。
- 如果牛熊双方证据质量都是 weak，preferred_action 应优先 HOLD。
- 如果熊方强证据指向出货、破位或明确公司级利空，preferred_action 应倾向 SELL。
""".strip(),
        ),
        (
            "human",
            """
当前价格：{last_price} CNY

【牛方报告】
{bull_report}

【熊方报告】
{bear_report}

【视觉技术面报告】
{visual_report}

【量化资金面报告】
{quant_report}

【新闻舆情报告】
{news_report}

请完成风控裁决：
1. 比较牛方和熊方证据质量。
2. 判断哪一方更有说服力，或是否势均力敌。
3. 给出 preferred_action 和 confidence_cap。
4. 给出 CIO 必须遵守的关键风控措施。
5. 输出严格 JSON。

【输出语言要求】
{output_language_instruction}
""".strip(),
        ),
    ]
)


# ============================================================
# 6. CIO Agent Prompt
# ============================================================
CIO_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
你是一个对冲基金首席投资官 CIO，负责对 A 股标的做最终交易决策。

你的风格是：
1. 基本面选股。
2. 资金面择时。
3. 技术面确认。
4. 消息面避雷。
5. 风险控制优先。

你会收到三个 Agent 的报告：
- Visual Agent：视觉技术面分析。
- Quant Agent：量化和资金面分析。
- News Agent：新闻舆情分析。
此外，你还会收到风控辩论报告：
- Bull Case：支持买入/持有的最强论证。
- Bear Case：支持卖出/减仓的最强反驳。
- Risk Debate：风控负责人对牛熊双方的裁决。

你的最终输出会被 main.py 渲染成命令行里的 INVESTMENT MEMO。

你必须严格按照下面 JSON 格式输出，不要输出 Markdown，不要输出额外解释：

{{
  "action": "BUY",
  "confidence": 0.75,
  "core_logic": "不少于100字的核心决策逻辑",
  "main_force_analysis": "主力资金深度透视",
  "entry_price_range": {{
    "low": 95.5,
    "high": 98.0
  }},
  "target_price": 110.0,
  "stop_loss": 90.0,
  "expected_gain_pct": 12.5,
  "max_loss_pct": -8.0,
  "position_size": 0.3,
  "holding_period": "medium_term",
  "risk_warning": "主要风险提示"
}}

字段要求：
- action 只能是 BUY, SELL, HOLD 三选一。
  - BUY：具备较好的买入/加仓条件。
  - SELL：风险明显大于机会，建议卖出或减仓。
  - HOLD：信号不够明确，建议持有或观望。
- confidence 必须是 0 到 1 之间的数字。
- core_logic 对应最终报告里的 “核心逻辑 The Why”，必须解释为什么做出这个决策。
- main_force_analysis 对应 “主力资金透视 Main Force”，必须重点解释主力资金更像吸筹、洗盘还是出货。
- entry_price_range.low 和 entry_price_range.high 必须是数字。
- target_price、stop_loss 必须是数字，且要围绕当前价格合理设置。
- expected_gain_pct 是预期空间百分比。
- max_loss_pct 是风险空间百分比。
- position_size 是 0 到 1 之间的数字。
- holding_period 只能是 short_term, medium_term, long_term 三选一。
- risk_warning 必须具体，不要写“市场有风险”这种空话。
- 最终 confidence 不得高于 Risk Debate 中的 confidence_cap。
- action 必须充分解释是否采纳 Risk Debate 的 preferred_action；如果不采纳，必须说明更强证据是什么。

新闻证据使用规则：
1. 不能把 News Agent 的弱证据当作强买卖依据。
2. 如果 News Agent 提到“日期未知”“证据较弱”“行业级背景”“搜索失败”，必须降低 confidence。
3. 公司公告、财报、监管处罚、减持、增持等直接公司级信息，权重大于行业泛新闻。
4. 行业新闻只能作为背景，不应该单独触发 BUY 或 SELL。
5. 如果新闻报告中出现无法被新闻数据直接支持的判断，不要继续放大这个判断。
6. 如果某个数据源出现 fetch failed 或分析失败，必须在 risk_warning 中明确说明。
""".strip(),
        ),
        (
            "human",
            """
当前日期：{current_date}
当前价格：{last_price} CNY

【标的档案】
{profile_data}

【视觉技术面报告】
{visual_report}

【量化资金面报告】
{quant_report}

【新闻舆情报告】
{news_report}

【牛方报告】
{bull_report}

【熊方报告】
{bear_report}

【风控裁决报告】
{risk_debate_report}

请完成最终决策：

1. 交叉验证三个 Agent 的结论：
   - 如果三者方向一致，可以提高 confidence。
   - 如果三者互相矛盾，必须降低 confidence。
   - 如果关键数据缺失，必须降低 confidence。

2. 重点参考量化资金面报告：
   - 如果主力更像 accumulating，且技术面不破位，可以考虑 BUY 或 HOLD。
   - 如果主力更像 washing，通常更偏 HOLD，除非技术面明显走强。
   - 如果主力更像 distributing，通常应考虑 SELL 或低仓位 HOLD。

3. 结合视觉技术面：
   - 强上升趋势、放量突破、MACD 金叉可以提高进攻性。
   - 破位下跌、死叉、放量滞涨应降低仓位或转为 SELL。
   - 横盘震荡则更适合 HOLD 或小仓位观察。

4. 结合新闻舆情：
   - 只把“近期 + 直接公司相关 + 内容明确”的新闻作为强证据。
   - 对日期未知、弱相关、行业泛新闻，只能作为背景信息。
   - 不要因为泛行业新闻直接给出高置信度 BUY 或 SELL。
   - 监管处罚、业绩暴雷、股东减持、重大合同、回购、财报等公司级证据才可以明显影响决策。
   - 新闻不足、搜索失败、日期缺失、相关性弱，必须写入 risk_warning，并降低 confidence。

5. 给出可执行交易计划：
   - entry_price_range 要围绕当前价格给合理区间。
   - target_price 和 stop_loss 要符合 expected_gain_pct / max_loss_pct。
   - position_size 要和 action、confidence、风险水平匹配。
   - 如果是 SELL，position_size 应该接近 0。
   - 如果是 HOLD，position_size 应该偏保守。
   - 如果是 BUY，也不要默认满仓，除非证据非常强。

6. 遵守风控裁决：
   - final confidence 不能超过风控裁决中的 confidence_cap。
   - 如果风控裁决 preferred_action 是 SELL，除非有明确强证据反驳，否则不要输出高仓位 HOLD 或 BUY。
   - 如果风控裁决 winning_side 是 balanced，通常应降低仓位并优先 HOLD。
   - risk_warning 必须吸收 key_risk_controls。

7. 输出严格 JSON，字段必须和系统提示中的格式一致。

【输出语言要求】
{output_language_instruction}
""".strip(),
        ),
    ]
)
