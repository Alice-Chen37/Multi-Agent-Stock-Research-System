"""
LangGraph 工作流定义

整体流程：

用户输入股票代码
    ↓
fetcher_node：抓取行情、K线图、Prophet预测、新闻、资金流、公司档案
    ↓
并行分析：
    quant_node：量化/资金面分析
    visual_node：K线图视觉分析
    news_node：新闻/舆情分析
    ↓
投研辩论：
    bull_node：构建买入/持有的最强论证
    bear_node：针对牛方报告提出卖出/减仓反驳
    risk_node：风控裁决牛熊分歧并给出置信度上限
    ↓
cio_node：综合三个 Agent 的报告，生成最终投资决策
    ↓
END
"""

from __future__ import annotations

import os
import asyncio
import ast
import json
import re
from datetime import datetime
from typing import NotRequired, TypedDict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=True)
from langchain.chat_models import init_chat_model
from langgraph.graph import END, StateGraph

from src.agents import (
    BEAR_PROMPT,
    BULL_PROMPT,
    CIO_PROMPT,
    NEWS_PROMPT,
    QUANT_PROMPT,
    RISK_PROMPT,
    VisualAgent,
    extract_json_from_text,
)
from src.models import (
    BearCaseReport,
    BullCaseReport,
    FinalDecision,
    NewsReport,
    QuantReport,
    RiskDebateReport,
    VisualReport,
)
from src.settings import is_english, output_language_instruction
from src.tools import (
    forecast_with_prophet,
    generate_stock_chart,
    get_money_flow,
    get_stock_data,
    get_stock_profile,
    search_news,
)




# ============================================================
# 1. LLM 初始化 / Multi-Free-Model Router
# ============================================================
import random


def _model_pool(env_name: str, default: str) -> list[str]:
    raw = os.getenv(env_name, default)
    models = [item.strip() for item in raw.split(",") if item.strip()]
    # 去重但保持顺序
    return list(dict.fromkeys(models))


DEFAULT_TEXT_POOL = (
    # openrouter/free is the $0 primary router. Gemma entries remain only as last-resort
    # explicit free fallbacks; provider 429s are skipped immediately.
    "openrouter/free,"
    "google/gemma-4-31b-it:free,"
    "google/gemma-4-26b-a4b-it:free"
)

AGENT_MODEL_POOLS = {
    "Quant": _model_pool("QUANT_MODELS", DEFAULT_TEXT_POOL),
    "News": _model_pool("NEWS_MODELS", DEFAULT_TEXT_POOL),
    "Bull": _model_pool("DEBATE_MODELS", DEFAULT_TEXT_POOL),
    "Bear": _model_pool("DEBATE_MODELS", DEFAULT_TEXT_POOL),
    "Risk": _model_pool("DEBATE_MODELS", DEFAULT_TEXT_POOL),
    "CIO": _model_pool("CIO_MODELS", DEFAULT_TEXT_POOL),
}

LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "90"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "1"))
LLM_RETRY_BASE_SECONDS = float(os.getenv("LLM_RETRY_BASE_SECONDS", "4"))
LLM_RETRY_MAX_SECONDS = float(os.getenv("LLM_RETRY_MAX_SECONDS", "12"))
LLM_MAX_CONCURRENCY = max(1, int(os.getenv("LLM_MAX_CONCURRENCY", "1")))
# OpenRouter free tier can enforce a global requests-per-minute cap.  When that
# specific cap is hit, switching models only burns more requests, so pause once
# and retry the same model after the window has had time to reset.
FREE_TIER_COOLDOWN_SECONDS = float(os.getenv("FREE_TIER_COOLDOWN_SECONDS", "65"))
FREE_TIER_MAX_WAITS = max(0, int(os.getenv("FREE_TIER_MAX_WAITS", "1")))
_llm_semaphore = asyncio.Semaphore(LLM_MAX_CONCURRENCY)
_llm_cache = {}


def _get_llm(model_name: str):
    if model_name not in _llm_cache:
        _llm_cache[model_name] = init_chat_model(
            model=model_name,
            model_provider="openai",
            temperature=0.2,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
    return _llm_cache[model_name]


def _is_daily_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "free-models-per-day" in msg or "openrouter_free_tier_daily" in msg


def _is_auth_error(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 401 or "missing authentication header" in str(exc).lower()


def _is_rate_limit_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) == 429:
        return True
    text = str(exc).lower()
    return "429" in text or "rate limit" in text or "rate-limited" in text


def _is_openrouter_free_tier_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "free-models-per-min" in text
        or "openrouter_free_tier_per_minute" in text
        or ("x-ratelimit-remaining" in text and "'0'" in text)
    )


def _retry_delay(attempt: int) -> float:
    base = min(LLM_RETRY_MAX_SECONDS, LLM_RETRY_BASE_SECONDS * (2 ** attempt))
    return base + random.uniform(0.0, min(1.5, base * 0.25))


def _response_text(response) -> str:
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if value:
                    parts.append(str(value))
            else:
                value = getattr(item, "text", None)
                parts.append(str(value if value is not None else item))
        return "\n".join(parts).strip()
    return ""


def _parse_json_robust(content: str) -> dict:
    """Parse common LLM JSON imperfections without making another API call."""
    try:
        return extract_json_from_text(content)
    except Exception as first_error:
        text = content.strip()
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise first_error
        candidate = text[start:end + 1]
        # Remove trailing commas before closing braces/brackets.
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(candidate)
        except Exception:
            pass
        # Some free models emit Python-style dicts (single quotes/True/None).
        try:
            value = ast.literal_eval(candidate)
            if isinstance(value, dict):
                return value
        except Exception:
            pass
        raise first_error


async def invoke_structured(messages, model_cls, label: str):
    """
    V3 free LLM-first structured router.

    Primary path:
    - Ask OpenRouter/provider for native JSON Schema structured output.
    - require_parameters=True prevents routing to endpoints that ignore response_format.
    - Validate again locally with Pydantic.

    Recovery path:
    - Locally repair common JSON imperfections without another API request.
    - Provider-specific 429 -> switch model.
    - Account-wide free-tier RPM -> wait once, retry same model.
    - Deterministic node fallbacks remain the final safety net only.
    """
    pool = AGENT_MODEL_POOLS.get(label) or _model_pool(
        "OPENAI_MODELS", DEFAULT_TEXT_POOL
    )
    failures: list[str] = []

    # Pydantic v2 JSON schema. OpenRouter uses the OpenAI response_format shape.
    schema = model_cls.model_json_schema()
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": f"{label.lower()}_report",
            "strict": True,
            "schema": schema,
        },
    }

    for model_index, model_name in enumerate(pool, start=1):
        print(
            f"[StructuredLLM] {label}: trying {model_name} "
            f"({model_index}/{len(pool)}) with strict JSON schema"
        )
        llm = _get_llm(model_name)
        free_tier_waits = 0

        while True:
            try:
                # These kwargs are forwarded through the OpenAI-compatible client.
                # OpenRouter's provider.require_parameters keeps the request on an
                # endpoint that supports response_format/json_schema.
                structured_llm = llm.bind(
                    response_format=response_format,
                    extra_body={"provider": {"require_parameters": True}},
                )

                async with _llm_semaphore:
                    response = await asyncio.wait_for(
                        structured_llm.ainvoke(messages),
                        timeout=LLM_TIMEOUT_SECONDS,
                    )

                content = _response_text(response)
                if not content:
                    raise ValueError("empty model response")

                # Native structured output should already be valid JSON. Keep the
                # local parser as a zero-cost recovery layer for imperfect free endpoints.
                data = _parse_json_robust(content)
                validated = model_cls.model_validate(data)
                actual_model = getattr(response, "response_metadata", {}).get(
                    "model_name", model_name
                )
                print(f"[StructuredLLM] {label}: success via {actual_model}")
                return validated

            except Exception as exc:
                failures.append(f"{model_name}: {type(exc).__name__}: {exc}")
                if _is_auth_error(exc):
                    raise RuntimeError(f"{label}: OpenRouter authentication failed (401). Check client credentials.") from exc
                if _is_daily_limit_error(exc):
                    raise RuntimeError(f"{label}: OpenRouter free daily quota exhausted. No further model retries.") from exc

                if _is_openrouter_free_tier_limit(exc):
                    if free_tier_waits < FREE_TIER_MAX_WAITS:
                        free_tier_waits += 1
                        print(
                            f"[RateLimit] {label}: OpenRouter free-tier RPM exhausted; "
                            f"waiting {FREE_TIER_COOLDOWN_SECONDS:.0f}s for reset "
                            f"({free_tier_waits}/{FREE_TIER_MAX_WAITS})..."
                        )
                        await asyncio.sleep(FREE_TIER_COOLDOWN_SECONDS)
                        continue
                    preview = str(exc).replace("\n", " ")[:240]
                    print(
                        f"[StructuredLLM] {label}: free-tier limit still active "
                        f"({preview}); switching model..."
                    )
                    break

                if _is_rate_limit_error(exc):
                    preview = str(exc).replace("\n", " ")[:240]
                    print(
                        f"[RateLimit] {label}/{model_name}: provider rate-limited "
                        f"({preview}); switching model..."
                    )
                    break

                preview = str(exc).replace("\n", " ")[:240]
                print(
                    f"[StructuredLLM] {label}: {model_name} failed ({preview}); "
                    "switching model..."
                )
                break

    raise RuntimeError(
        f"{label}: all free structured-output models failed. Last errors: "
        + " | ".join(failures[-3:])
    )


# 视觉模型单独走 VisualAgent；它也有自己的免费多模型池。
visual_agent = VisualAgent()


# ============================================================
# 2. LangGraph State 定义
# ============================================================
class AgentState(TypedDict):
    """
    LangGraph 全局状态。

    注意：
    - 初始输入通常只有 symbol。
    - 后续字段由 fetcher / quant / visual / news / cio 节点逐步写入。
    - 所以除 symbol 外，其余字段使用 NotRequired。
    """

    symbol: str

    price_data: NotRequired[object]  # pandas DataFrame
    chart_path: NotRequired[str]
    prophet_result: NotRequired[str]
    news_data: NotRequired[str]
    fund_flow_data: NotRequired[str]
    profile_data: NotRequired[str]
    data_error: NotRequired[str]

    quant_report: NotRequired[QuantReport | None]
    visual_report: NotRequired[VisualReport | str | None]
    news_report: NotRequired[NewsReport | None]
    bull_report: NotRequired[BullCaseReport | None]
    bear_report: NotRequired[BearCaseReport | None]
    risk_debate_report: NotRequired[RiskDebateReport | None]
    final_report: NotRequired[FinalDecision | None]


# ============================================================
# 3. 工具安全调用函数
# ============================================================
def safe_tool_invoke(tool, payload: dict, default=None, label: str = "tool"):
    """
    安全调用 LangChain @tool 工具。

    这样做的原因：
    A股数据接口、新闻搜索接口、资金流接口都可能临时失败。
    如果一个工具失败就让整个工作流崩溃，用户体验会很差。

    所以这里统一 catch exception，并返回 default。
    """
    try:
        return tool.invoke(payload)
    except Exception as e:
        print(f"[Warning] {label} failed: {e}")
        return default


def log_text(cn: str, en: str) -> str:
    """
    Return localized log text for console progress messages.
    """
    return en if is_english() else cn


def safe_model_json(obj) -> str:
    """
    将 Pydantic 对象安全转换为 JSON 字符串。
    """
    if obj is None:
        return "N/A"

    if hasattr(obj, "model_dump_json"):
        try:
            return obj.model_dump_json(indent=2)
        except Exception:
            return str(obj)

    return str(obj)


def get_last_price_text(state: AgentState) -> str:
    """
    从 state 中提取最新收盘价文本，供多个 Agent 复用。
    """
    df = state.get("price_data")

    try:
        if df is not None and not getattr(df, "empty", True):
            return f"{float(df.iloc[-1]['close']):.2f}"
    except Exception:
        pass

    return "0.00"


def visual_report_to_text(visual_report) -> str:
    """
    将视觉报告统一转成 prompt 可读文本。
    """
    if isinstance(visual_report, VisualReport):
        return visual_report.model_dump_json(indent=2)

    return str(visual_report or "N/A")


def build_local_visual_report(df) -> VisualReport:
    """Remote vision unavailable 时，用真实 OHLC/指标生成保守的本地技术报告。"""
    if df is None or getattr(df, "empty", True):
        raise ValueError("price_data is empty")

    recent = df.tail(min(20, len(df))).copy()
    close = recent["close"].astype(float)
    last = float(close.iloc[-1])

    # 支撑/压力只来自最近真实 K 线，不人为按百分比制造。
    if "low" in recent.columns:
        support = float(recent["low"].astype(float).min())
    else:
        support = float(close.min())
    if "high" in recent.columns:
        resistance = float(recent["high"].astype(float).max())
    else:
        resistance = float(close.max())

    # 5/10 日均价关系 + 近 5 日变化，做保守趋势分类。
    ma5 = float(close.tail(min(5, len(close))).mean())
    ma10 = float(close.tail(min(10, len(close))).mean())
    lookback = min(5, len(close) - 1)
    change = 0.0 if lookback <= 0 else last / float(close.iloc[-1 - lookback]) - 1.0
    if change >= 0.05 and last > ma5 >= ma10:
        trend = "strong_uptrend"
    elif change > 0.01 and last >= ma5:
        trend = "weak_uptrend"
    elif change <= -0.05 and last < ma5 <= ma10:
        trend = "strong_downtrend"
    elif change < -0.01 and last <= ma5:
        trend = "weak_downtrend"
    else:
        trend = "sideways"

    # MACD：优先使用数据源已经计算好的 macd 列；仅判断零轴/最近方向。
    macd_signal = "neutral"
    macd_note = "MACD 数据不足"
    if "macd" in recent.columns:
        m = recent["macd"].dropna().astype(float)
        if len(m) >= 2:
            prev_m, curr_m = float(m.iloc[-2]), float(m.iloc[-1])
            if prev_m <= 0 < curr_m:
                macd_signal = "golden_cross"
            elif prev_m >= 0 > curr_m:
                macd_signal = "death_cross"
            macd_note = f"MACD {prev_m:.3f}→{curr_m:.3f}"

    patterns: list[str] = ["本地OHLC回退分析"]
    # 只识别简单、可由 OHLC 明确计算的形态。
    if all(c in recent.columns for c in ("open", "high", "low", "close")):
        row = recent.iloc[-1]
        o, h, l, c = map(float, (row["open"], row["high"], row["low"], row["close"]))
        span = max(h - l, 1e-9)
        body = abs(c - o)
        if body / span <= 0.10:
            patterns.append("十字/小实体K线")
        lower_shadow = min(o, c) - l
        upper_shadow = h - max(o, c)
        if lower_shadow >= 2 * max(body, 1e-9) and upper_shadow <= max(body, span * 0.15):
            patterns.append("长下影线")
        elif upper_shadow >= 2 * max(body, 1e-9) and lower_shadow <= max(body, span * 0.15):
            patterns.append("长上影线")

    volume_note = "成交量数据不足"
    if "volume" in recent.columns:
        v = recent["volume"].dropna().astype(float)
        if len(v) >= 6:
            avg5 = float(v.iloc[-6:-1].mean())
            if avg5 > 0:
                ratio = float(v.iloc[-1]) / avg5
                volume_note = f"最新成交量约为前5日均量的 {ratio:.2f} 倍"

    rsi_note = ""
    if "rsi" in recent.columns:
        r = recent["rsi"].dropna()
        if len(r):
            rsi_note = f"，RSI={float(r.iloc[-1]):.2f}"

    conclusion = (
        f"远程视觉模型不可用，系统已自动切换到本地 OHLC 技术分析。"
        f"最新收盘价 {last:.2f}，近20个交易日支撑约 {support:.2f}、压力约 {resistance:.2f}；"
        f"趋势分类为 {trend}，{macd_note}{rsi_note}，{volume_note}。"
        "该回退报告基于结构化行情计算，不等同于图像模型对趋势线或复杂K线组合的视觉识别。"
    )
    return VisualReport(
        key_patterns=patterns[:5],
        macd_signal=macd_signal,
        trend_direction=trend,
        support_resistance={"support": round(support, 2), "resistance": round(resistance, 2)},
        visual_conclusion=conclusion,
    )


# ============================================================
# 4. 节点定义：Fetcher
# ============================================================
def fetcher_node(state: AgentState):
    """
    数据中心节点。

    负责统一抓取：
    1. 股票行情数据
    2. K线图图片路径
    3. Prophet 时间序列预测结果
    4. 新闻数据
    5. 资金流数据
    6. 公司档案数据
    """
    symbol = state["symbol"]
    print(
        "\n"
        + log_text(
            f"[System] 正在获取 {symbol} 的数据...",
            f"[System] Fetching data for {symbol}...",
        )
    )

    df = safe_tool_invoke(
        get_stock_data,
        {"symbol": symbol},
        default=None,
        label="get_stock_data",
    )

    if df is None or getattr(df, "empty", True):
        print(
            log_text(
                f"[Warning] 未找到 {symbol} 的行情数据。",
                f"[Warning] No price data found for {symbol}.",
            )
        )
        return {
            "price_data": None,
            "chart_path": "",
            "prophet_result": "No price data available.",
            "news_data": "No price data available.",
            "fund_flow_data": "No price data available.",
            "profile_data": f"Symbol: {symbol}\nProfile fetch skipped because price data is empty.",
            "data_error": f"No market price data available for {symbol}.",
        }

    chart_path = safe_tool_invoke(
        generate_stock_chart,
        {"df": df, "symbol": symbol},
        default="",
        label="generate_stock_chart",
    )

    prophet_res = safe_tool_invoke(
        forecast_with_prophet,
        {"df": df},
        default="Prophet forecast failed.",
        label="forecast_with_prophet",
    )

    news = safe_tool_invoke(
        search_news,
        {"symbol": symbol},
        default="News fetch failed.",
        label="search_news",
    )

    flow = safe_tool_invoke(
        get_money_flow,
        {"symbol": symbol},
        default="Money flow fetch failed.",
        label="get_money_flow",
    )

    profile = safe_tool_invoke(
        get_stock_profile,
        {"symbol": symbol},
        default=f"Symbol: {symbol}\nCompany profile fetch failed.",
        label="get_stock_profile",
    )

    return {
        "price_data": df,
        "chart_path": chart_path,
        "prophet_result": prophet_res,
        "news_data": news,
        "fund_flow_data": flow,
        "profile_data": profile,
        "data_error": "",
    }


# ============================================================
# 5. 节点定义：Quant Agent
# ============================================================
async def quant_node(state: AgentState):
    """
    量化分析节点。

    输入：
    - price_data
    - prophet_result
    - fund_flow_data

    输出：
    - quant_report: QuantReport
    """
    print(
        log_text(
            "[Agent] Quant Analyst 正在计算...",
            "[Agent] Quant Analyst is calculating...",
        )
    )

    df = state.get("price_data")

    if df is None or getattr(df, "empty", True):
        return {
            "quant_report": QuantReport(
                main_force_intent="washing",
                confidence_score=0.0,
                key_technical_signals=["无行情数据"],
                price_volume_analysis="无法获取行情数据，量价分析不可用。",
                risk_assessment="high",
                detailed_analysis="由于没有有效的行情数据，量化模型无法判断主力资金意图。本次量化报告仅作为失败占位，不应作为真实交易依据。",
            )
        }

    try:
        needed_cols = ["date", "close", "rsi", "macd"]
        available_cols = [col for col in needed_cols if col in df.columns]

        if available_cols:
            recent_str = df.tail(5)[available_cols].to_string()
        else:
            recent_str = df.tail(5).to_string()

        messages = QUANT_PROMPT.invoke(
            {
                "prophet_forecast": state.get("prophet_result", "N/A"),
                "market_data_str": recent_str,
                "fund_flow_data": state.get("fund_flow_data", "N/A"),
                "output_language_instruction": output_language_instruction(),
            }
        )

        result = await invoke_structured(messages, QuantReport, "Quant")
        return {"quant_report": result}

    except Exception as e:
        print(f"[Warning] Quant structured output failed: {e}")

        fallback_report = QuantReport(
            main_force_intent="washing",
            confidence_score=0.3,
            key_technical_signals=["结构化分析失败"],
            price_volume_analysis="量化节点运行失败，可能是模型结构化输出格式不符合 Pydantic 要求。",
            risk_assessment="high",
            detailed_analysis=(
                "量化分析结构化输出失败。本次无法稳定判断主力是在吸筹、洗盘还是出货。"
                "建议检查 OPENAI_MODEL_NAME、OPENAI_BASE_URL、OPENAI_API_KEY 是否正确，"
                "以及 QUANT_PROMPT 是否与 QuantReport 字段保持一致。"
            ),
        )

        return {"quant_report": fallback_report}


# ============================================================
# 6. 节点定义：Visual Agent
# ============================================================
async def visual_node(state: AgentState):
    """
    视觉图表分析节点。

    输入：
    - chart_path

    输出：
    - visual_report: VisualReport | str
    """
    print(
        log_text(
            "[Agent] Visual Analyst 正在查看 K 线图...",
            "[Agent] Visual Analyst is looking at the chart...",
        )
    )

    chart_path = state.get("chart_path", "")

    if not chart_path:
        return {"visual_report": "No Chart Image Available"}

    try:
        report = await asyncio.to_thread(visual_agent.analyze_chart, chart_path)
        if isinstance(report, VisualReport):
            return {"visual_report": report}

        # Remote vision returned a failure string: degrade to deterministic local OHLC analysis.
        print(f"[Warning] Remote Visual unavailable: {report}")
        print("[Fallback] Visual: running local OHLC technical analysis...")
        local_report = build_local_visual_report(state.get("price_data"))
        print("[Fallback] Visual: local OHLC technical analysis succeeded.")
        return {"visual_report": local_report}

    except Exception as e:
        print(f"[Warning] Visual analysis failed: {e}")
        try:
            print("[Fallback] Visual: running local OHLC technical analysis...")
            local_report = build_local_visual_report(state.get("price_data"))
            print("[Fallback] Visual: local OHLC technical analysis succeeded.")
            return {"visual_report": local_report}
        except Exception as fallback_error:
            print(f"[Warning] Local Visual fallback failed: {fallback_error}")
            return {"visual_report": f"Visual Analysis Failed: {e}; local fallback failed: {fallback_error}"}


# ============================================================
# 7. 节点定义：News Agent
# ============================================================
async def news_node(state: AgentState):
    """
    新闻舆情分析节点。

    输入：
    - news_data

    输出：
    - news_report: NewsReport
    """
    print(
        log_text(
            "[Agent] News Analyst 正在阅读新闻...",
            "[Agent] News Analyst is reading...",
        )
    )

    try:
        messages = NEWS_PROMPT.invoke(
            {
                "news_data": state.get("news_data", "N/A"),
                "output_language_instruction": output_language_instruction(),
            }
        )

        result = await invoke_structured(messages, NewsReport, "News")
        return {"news_report": result}

    except Exception as e:
        print(f"[Warning] News structured output failed: {e}")
        print("[Info] Falling back to basic NewsReport...")

        fallback_report = NewsReport(
            company_profile="新闻结构化分析失败，无法提取完整公司档案。",
            sentiment_score=0.0,
            key_news_summary=["新闻结构化分析失败"],
            market_positioning="N/A",
            risk_warnings=["新闻数据解析错误或模型输出格式不符合要求"],
        )

        return {"news_report": fallback_report}


# ============================================================
# 8. 节点定义：Bull / Bear / Risk Debate Agents
# ============================================================
async def bull_node(state: AgentState):
    """
    牛方节点。

    输入：
    - quant_report
    - visual_report
    - news_report
    - profile_data

    输出：
    - bull_report: BullCaseReport
    """
    print(
        log_text(
            "[Agent] Bull Researcher 正在构建最强持有/买入论证...",
            "[Agent] Bull Researcher is building the strongest hold/buy case...",
        )
    )

    try:
        messages = BULL_PROMPT.invoke(
            {
                "last_price": get_last_price_text(state),
                "profile_data": state.get("profile_data", "N/A"),
                "visual_report": visual_report_to_text(state.get("visual_report")),
                "quant_report": safe_model_json(state.get("quant_report")),
                "news_report": safe_model_json(state.get("news_report")),
                "output_language_instruction": output_language_instruction(),
            }
        )

        result = await invoke_structured(messages, BullCaseReport, "Bull")
        return {"bull_report": result}

    except Exception as e:
        print(f"[Warning] Bull structured output failed: {e}")
        return {
            "bull_report": BullCaseReport(
                stance="HOLD",
                strongest_points=["牛方结构化分析失败，无法形成强看多证据"],
                evidence_quality="weak",
                counter_to_bear_risks="由于牛方节点失败，无法有效反驳潜在看空风险。",
                invalidation_condition="若关键行情、资金流或新闻数据仍无法验证，应视为牛方观点无效。",
            )
        }


async def bear_node(state: AgentState):
    """
    熊方节点。

    输入：
    - bull_report
    - quant_report
    - visual_report
    - news_report
    - profile_data

    输出：
    - bear_report: BearCaseReport
    """
    print(
        log_text(
            "[Agent] Bear Researcher 正在反驳牛方论证...",
            "[Agent] Bear Researcher is challenging the bull case...",
        )
    )

    try:
        messages = BEAR_PROMPT.invoke(
            {
                "last_price": get_last_price_text(state),
                "profile_data": state.get("profile_data", "N/A"),
                "visual_report": visual_report_to_text(state.get("visual_report")),
                "quant_report": safe_model_json(state.get("quant_report")),
                "news_report": safe_model_json(state.get("news_report")),
                "bull_report": safe_model_json(state.get("bull_report")),
                "output_language_instruction": output_language_instruction(),
            }
        )

        result = await invoke_structured(messages, BearCaseReport, "Bear")
        return {"bear_report": result}

    except Exception as e:
        print(f"[Warning] Bear structured output failed: {e}")
        return {
            "bear_report": BearCaseReport(
                stance="HOLD",
                strongest_points=["熊方结构化分析失败，无法形成可靠卖出证据"],
                evidence_quality="weak",
                rebuttal_to_bull_case="由于熊方节点失败，无法对牛方报告做有效逐点反驳。",
                risk_trigger="若价格放量跌破关键支撑或资金连续流出，应重新评估减仓。",
            )
        }


async def risk_node(state: AgentState):
    """
    风控裁决节点。

    输入：
    - bull_report
    - bear_report
    - quant_report
    - visual_report
    - news_report

    输出：
    - risk_debate_report: RiskDebateReport
    """
    print(
        log_text(
            "[Agent] Risk Manager 正在裁决牛熊辩论...",
            "[Agent] Risk Manager is judging the bull/bear debate...",
        )
    )

    try:
        messages = RISK_PROMPT.invoke(
            {
                "last_price": get_last_price_text(state),
                "bull_report": safe_model_json(state.get("bull_report")),
                "bear_report": safe_model_json(state.get("bear_report")),
                "visual_report": visual_report_to_text(state.get("visual_report")),
                "quant_report": safe_model_json(state.get("quant_report")),
                "news_report": safe_model_json(state.get("news_report")),
                "output_language_instruction": output_language_instruction(),
            }
        )

        result = await invoke_structured(messages, RiskDebateReport, "Risk")
        return {"risk_debate_report": result}

    except Exception as e:
        print(f"[Warning] Risk structured output failed: {e}")
        return {
            "risk_debate_report": RiskDebateReport(
                preferred_action="HOLD",
                confidence_cap=0.5,
                winning_side="balanced",
                key_risk_controls=["风控节点失败，最终决策置信度不得超过 50%"],
                debate_summary=(
                    "风控裁决结构化输出失败，无法可靠比较牛熊双方证据。"
                    "CIO 应优先保持保守，避免高仓位进攻。"
                ),
            )
        }


def build_deterministic_cio(state: AgentState, failure_reason: str = "") -> FinalDecision:
    """
    Last-resort CIO built only from already validated upstream reports and real OHLC levels.
    It does not call another model and never invents fixed +/- percentage price levels.
    """
    try:
        current_price = float(get_last_price_text(state))
        if current_price <= 0:
            raise ValueError("invalid current price")
    except Exception:
        current_price = 0.0

    support = current_price
    resistance = current_price
    visual = state.get("visual_report")
    if isinstance(visual, VisualReport):
        try:
            sr = visual.support_resistance
            if hasattr(sr, "model_dump"):
                sr = sr.model_dump()
            if isinstance(sr, dict):
                support = float(sr.get("support") or current_price)
                resistance = float(sr.get("resistance") or current_price)
            else:
                support = float(getattr(sr, "support", current_price) or current_price)
                resistance = float(getattr(sr, "resistance", current_price) or current_price)
        except Exception:
            support = resistance = current_price

    risk = state.get("risk_debate_report")
    preferred = str(getattr(risk, "preferred_action", "HOLD") or "HOLD").upper()
    if preferred not in {"BUY", "SELL", "HOLD"}:
        preferred = "HOLD"
    cap = float(getattr(risk, "confidence_cap", 0.35) or 0.35)
    cap = max(0.0, min(cap, 1.0))

    quant = state.get("quant_report")
    q_conf = float(getattr(quant, "confidence_score", 0.0) or 0.0)
    q_conf = max(0.0, min(q_conf, 1.0))
    intent = str(getattr(quant, "main_force_intent", "unknown") or "unknown")

    # Deterministic confidence is intentionally conservative and can never exceed Risk.
    confidence = min(cap, 0.45, max(0.15, q_conf * 0.60))
    action = preferred
    position = 0.0
    if action == "BUY":
        position = min(0.20, max(0.05, confidence * 0.30))
    elif action == "HOLD":
        position = min(0.10, max(0.0, confidence * 0.15))

    expected_gain = ((resistance / current_price) - 1.0) * 100.0 if current_price > 0 else 0.0
    max_loss = ((support / current_price) - 1.0) * 100.0 if current_price > 0 else 0.0

    bull = state.get("bull_report")
    bear = state.get("bear_report")
    winning = str(getattr(risk, "winning_side", "balanced") or "balanced")
    reason = failure_reason.replace("\n", " ")[:180]

    return FinalDecision(
        action=action,
        confidence=round(confidence, 2),
        core_logic=(
            f"远程 CIO 未能产生可验证的结构化输出，系统已启用确定性 CIO。"
            f"最终动作直接遵守 Risk Debate 的 preferred_action={action}，winning_side={winning}，"
            f"且置信度严格限制在 confidence_cap={cap:.2f} 以下。"
            f"价格区间仅使用真实 OHLC 技术报告中的支撑/压力，不使用固定百分比推测。"
        ),
        main_force_analysis=(
            f"量化报告识别的主力状态为 {intent}，量化置信度约 {q_conf:.2f}。"
            "确定性 CIO 不重新推断或补造资金流事实。"
        ),
        entry_price_range={"low": current_price, "high": current_price},
        target_price=round(resistance, 2),
        stop_loss=round(support, 2),
        expected_gain_pct=round(expected_gain, 2),
        max_loss_pct=round(max_loss, 2),
        position_size=round(position, 2),
        holding_period="short_term",
        risk_warning=(
            "远程 CIO 结构化输出失败，本结果由已验证的 Quant/Visual/Bull/Bear/Risk 报告确定性合成；"
            "不应把该回退结果视为新的模型预测。" + (f" 原始错误：{reason}" if reason else "")
        ),
    )


def apply_cio_numerical_guardrails(state: AgentState, decision: FinalDecision) -> FinalDecision:
    """Ground CIO trading levels in real OHLC instead of model-invented prices.

    The LLM still owns action, qualitative reasoning, confidence, position size,
    holding period, and risk wording.  This function owns executable price
    numbers: entry range, target, stop, expected gain and max loss.
    """
    df = state.get("price_data")
    if df is None or getattr(df, "empty", True):
        return decision

    try:
        recent = df.tail(min(20, len(df))).copy()
        current = float(recent.iloc[-1]["close"])
        support = float(recent["low"].astype(float).min()) if "low" in recent.columns else float(recent["close"].astype(float).min())
        resistance = float(recent["high"].astype(float).max()) if "high" in recent.columns else float(recent["close"].astype(float).max())
        if current <= 0:
            return decision

        # Never invent a percentage-based entry range.  BUY may stage between
        # observed support and current price; HOLD/SELL use current price only.
        action = str(decision.action).upper()
        if action == "BUY":
            entry_low = min(support, current)
            entry_high = current
        else:
            entry_low = current
            entry_high = current

        expected_gain = ((resistance / current) - 1.0) * 100.0
        max_loss = ((support / current) - 1.0) * 100.0

        # Enforce Risk's confidence cap even if the LLM ignored the prompt.
        confidence = float(decision.confidence)
        risk = state.get("risk_debate_report")
        if risk is not None:
            try:
                confidence = min(confidence, float(risk.confidence_cap))
            except Exception:
                pass
        confidence = max(0.0, min(confidence, 1.0))

        updates = {
            "confidence": round(confidence, 2),
            "entry_price_range": {"low": round(entry_low, 2), "high": round(entry_high, 2)},
            "target_price": round(resistance, 2),
            "stop_loss": round(support, 2),
            "expected_gain_pct": round(expected_gain, 2),
            "max_loss_pct": round(max_loss, 2),
        }
        grounded = decision.model_copy(update=updates)
        print(
            f"[Guardrail] CIO prices grounded to real 20-day OHLC: "
            f"current={current:.2f}, support={support:.2f}, resistance={resistance:.2f}"
        )
        return grounded
    except Exception as exc:
        print(f"[Warning] CIO numerical guardrail skipped: {exc}")
        return decision


# ============================================================
# 9. 节点定义：CIO Agent
# ============================================================
async def cio_node(state: AgentState):
    """
    CIO 最终决策节点。

    输入：
    - quant_report
    - visual_report
    - news_report
    - profile_data
    - price_data

    输出：
    - final_report: FinalDecision
    """
    print(
        log_text(
            "[Agent] CIO 正在生成最终决策...",
            "[Agent] CIO is making final decision...",
        )
    )

    last_price = get_last_price_text(state)
    visual_report_str = visual_report_to_text(state.get("visual_report"))
    quant_report_str = safe_model_json(state.get("quant_report"))
    news_report_str = safe_model_json(state.get("news_report"))
    bull_report_str = safe_model_json(state.get("bull_report"))
    bear_report_str = safe_model_json(state.get("bear_report"))
    risk_debate_report_str = safe_model_json(state.get("risk_debate_report"))

    try:
        messages = CIO_PROMPT.invoke(
            {
                "visual_report": visual_report_str,
                "quant_report": quant_report_str,
                "news_report": news_report_str,
                "bull_report": bull_report_str,
                "bear_report": bear_report_str,
                "risk_debate_report": risk_debate_report_str,
                "profile_data": state.get("profile_data", "N/A"),
                "current_date": datetime.now().strftime("%Y-%m-%d"),
                "last_price": last_price,
                "output_language_instruction": output_language_instruction(),
            }
        )

        result = await invoke_structured(messages, FinalDecision, "CIO")
        result = apply_cio_numerical_guardrails(state, result)
        return {"final_report": result}

    except Exception as e:
        print(f"[Warning] CIO structured output failed: {e}")
        print("[Fallback] CIO: building deterministic decision from validated upstream reports...")
        fallback_decision = build_deterministic_cio(state, str(e))
        print("[Fallback] CIO: deterministic decision succeeded.")
        return {"final_report": fallback_decision}



# ============================================================
# 10. 图构建：Graph Construction
# ============================================================
def route_after_fetch(state: AgentState) -> str:
    """行情数据失败时直接结束，避免用 0 元价格继续生成伪报告。"""
    if state.get("data_error"):
        return "stop"
    return "continue"


def fanout_node(state: AgentState):
    return {}


builder = StateGraph(AgentState)
builder.add_node("fetcher", fetcher_node)
builder.add_node("fanout", fanout_node)
builder.add_node("quant", quant_node)
builder.add_node("visual", visual_node)
builder.add_node("news", news_node)
builder.add_node("bull", bull_node)
builder.add_node("bear", bear_node)
builder.add_node("risk", risk_node)
builder.add_node("cio", cio_node)

builder.set_entry_point("fetcher")
builder.add_conditional_edges(
    "fetcher",
    route_after_fetch,
    {"continue": "fanout", "stop": END},
)
builder.add_edge("fanout", "quant")
builder.add_edge("fanout", "visual")
builder.add_edge("fanout", "news")
builder.add_edge(["quant", "visual", "news"], "bull")
builder.add_edge("bull", "bear")
builder.add_edge("bear", "risk")
builder.add_edge("risk", "cio")
builder.add_edge("cio", END)
app = builder.compile()
