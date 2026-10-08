"""
LangChain V1+ 结构化输出模型定义

这个文件只负责定义“数据结构”和“最终报告展示格式”。

核心作用：
1. 用 Pydantic BaseModel 限制 AI 输出格式。
2. 让 Quant / News / Visual / CIO 四个 Agent 的输出变成稳定结构。
3. 将 CIO 的最终结构化结果转换成命令行可展示的 Markdown 报告。
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src.settings import is_english


# ============================================================
# 1. Quant Agent 输出：量化分析报告
# ============================================================
class QuantReport(BaseModel):
    """量化分析结构化报告 - 由 Quant Agent 生成"""

    main_force_intent: Literal["accumulating", "washing", "distributing"] = Field(
        description=(
            "主力资金意图判断："
            "accumulating 表示吸筹，washing 表示洗盘，distributing 表示出货"
        )
    )

    confidence_score: float = Field(
        ge=0,
        le=1,
        description="判断置信度，范围 0-1",
    )

    key_technical_signals: list[str] = Field(
        min_length=1,
        max_length=5,
        description="关键技术信号列表，例如：['RSI超卖', 'MACD金叉', '放量突破']",
    )

    price_volume_analysis: str = Field(
        description="量价配合分析，例如：缩量上涨、放量滞涨、主力控盘等"
    )

    risk_assessment: Literal["low", "medium", "high"] = Field(
        description="风险等级评估：low / medium / high"
    )

    detailed_analysis: str = Field(
        description="详细的量化分析文本，建议不少于100字"
    )

    @field_validator("key_technical_signals", mode="before")
    @classmethod
    def normalize_key_technical_signals(cls, value):
        """
        防止模型偶尔把列表输出成字符串。
        """
        if value is None:
            return ["无明确技术信号"]

        if isinstance(value, str):
            return [value]

        return value


# ============================================================
# 2. News Agent 输出：舆情分析报告
# ============================================================
class NewsReport(BaseModel):
    """舆情分析结构化报告 - 由 News Agent 生成"""

    company_profile: str = Field(
        description="公司基本信息：名称、行业、板块、核心概念。必须是字符串，不是字典"
    )

    sentiment_score: float = Field(
        ge=-1,
        le=1,
        description="综合舆情评分，-1 表示极度负面，1 表示极度正面",
    )

    key_news_summary: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="关键新闻摘要列表，仅保留近期新闻",
    )

    market_positioning: str = Field(
        description="市场地位分析：是否龙头、是否有政策支持、竞争优势等"
    )

    risk_warnings: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="潜在风险点列表",
    )

    @field_validator("key_news_summary", "risk_warnings", mode="before")
    @classmethod
    def normalize_string_list(cls, value):
        """
        防止模型把列表字段输出成单个字符串。
        """
        if value is None:
            return []

        if isinstance(value, str):
            return [value]

        return value


# ============================================================
# 3. Visual Agent 输出：视觉图表分析报告
# ============================================================
class VisualReport(BaseModel):
    """视觉分析结构化报告 - 由 Visual Agent 生成"""

    key_patterns: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="识别到的关键K线形态，例如：['头肩顶', '早晨之星', '乌云盖顶']",
    )

    macd_signal: Literal[
        "bullish_divergence",
        "bearish_divergence",
        "golden_cross",
        "death_cross",
        "neutral",
    ] = Field(
        description=(
            "MACD 信号："
            "bullish_divergence 底背离，bearish_divergence 顶背离，"
            "golden_cross 金叉，death_cross 死叉，neutral 中性"
        )
    )

    trend_direction: Literal[
        "strong_uptrend",
        "weak_uptrend",
        "sideways",
        "weak_downtrend",
        "strong_downtrend",
    ] = Field(
        description="趋势方向判断"
    )

    support_resistance: dict[str, float] = Field(
        default_factory=dict,
        description="支撑位和压力位，格式：{'support': 价格, 'resistance': 价格}",
    )

    visual_conclusion: str = Field(
        description="视觉分析结论，建议不少于50字"
    )

    @field_validator("key_patterns", mode="before")
    @classmethod
    def normalize_key_patterns(cls, value):
        """
        防止模型把 K 线形态列表输出成字符串。
        """
        if value is None:
            return []

        if isinstance(value, str):
            return [value]

        return value

    @field_validator("support_resistance", mode="before")
    @classmethod
    def normalize_support_resistance(cls, value):
        """
        防止视觉模型返回 support/resistance 为空或字符串导致后续报错。
        """
        if value is None:
            return {"support": 0.0, "resistance": 0.0}

        if isinstance(value, dict):
            result = {}

            for key in ["support", "resistance"]:
                raw = value.get(key)

                try:
                    result[key] = float(raw)
                except (TypeError, ValueError):
                    result[key] = 0.0

            return result

        return {"support": 0.0, "resistance": 0.0}


# ============================================================
# 4. Bull / Bear / Risk Debate 输出：投研反驳机制
# ============================================================
class BullCaseReport(BaseModel):
    """牛方报告 - 支持买入、加仓或继续持有的最强论据"""

    stance: Literal["BUY", "HOLD"] = Field(
        description="牛方倾向，只能是 BUY 或 HOLD"
    )

    strongest_points: list[str] = Field(
        min_length=1,
        max_length=5,
        description="支持买入、加仓或继续持有的最强证据",
    )

    evidence_quality: Literal["strong", "medium", "weak"] = Field(
        description="牛方证据质量：strong / medium / weak"
    )

    counter_to_bear_risks: str = Field(
        description="预先回应潜在看空风险，说明为什么这些风险暂不足以触发卖出"
    )

    invalidation_condition: str = Field(
        description="牛方观点失效条件，例如跌破关键支撑、资金持续流出、利空兑现等"
    )


class BearCaseReport(BaseModel):
    """熊方报告 - 支持卖出、减仓或回避的最强论据"""

    stance: Literal["SELL", "HOLD"] = Field(
        description="熊方倾向，只能是 SELL 或 HOLD"
    )

    strongest_points: list[str] = Field(
        min_length=1,
        max_length=5,
        description="支持卖出、减仓或谨慎持有的最强风险证据",
    )

    evidence_quality: Literal["strong", "medium", "weak"] = Field(
        description="熊方证据质量：strong / medium / weak"
    )

    rebuttal_to_bull_case: str = Field(
        description="针对牛方报告的逐点反驳，必须引用牛方观点中的具体弱点"
    )

    risk_trigger: str = Field(
        description="触发减仓或卖出的关键条件，例如放量破位、主力出货、重大利空等"
    )


class RiskDebateReport(BaseModel):
    """风控裁决报告 - 综合牛熊双方后给 CIO 的风险建议"""

    preferred_action: Literal["BUY", "SELL", "HOLD"] = Field(
        description="风控后的倾向动作：BUY / SELL / HOLD"
    )

    confidence_cap: float = Field(
        ge=0,
        le=1,
        description="建议 CIO 最终置信度上限，数据缺失或牛熊分歧大时必须降低",
    )

    winning_side: Literal["bull", "bear", "balanced"] = Field(
        description="本轮论证中更有说服力的一方：bull / bear / balanced"
    )

    key_risk_controls: list[str] = Field(
        min_length=1,
        max_length=5,
        description="关键风控措施，例如仓位、止损、确认信号、禁买条件",
    )

    debate_summary: str = Field(
        description="总结牛熊双方核心分歧、证据强弱和最终风控判断"
    )


# ============================================================
# 5. CIO Agent 输出：最终投资决策
# ============================================================
class FinalDecision(BaseModel):
    """最终投资决策 - 由 CIO Agent 生成"""

    action: Literal["BUY", "SELL", "HOLD"] = Field(
        description="交易指令：BUY 买入，SELL 卖出，HOLD 持有/观望"
    )

    confidence: float = Field(
        ge=0,
        le=1,
        description="决策置信度，范围 0-1",
    )

    core_logic: str = Field(
        description="核心决策逻辑，详细阐述理由，建议不少于100字"
    )

    main_force_analysis: str = Field(
        description="主力资金深度透视，基于量化报告分析主力资金动向"
    )

    entry_price_range: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "价格操作区间，格式：{'low': 下限, 'high': 上限}。"
            "BUY 时表示进场区间；SELL 时表示减仓/离场区间；HOLD 时表示观察区间。"
        ),
    )

    target_price: float = Field(
        gt=0,
        description=(
            "目标价格。BUY 时表示止盈目标；SELL 时表示下方观察位；"
            "HOLD 时表示上方压力位。单位 CNY。"
        ),
    )

    stop_loss: float = Field(
        gt=0,
        description=(
            "风险价格。BUY 时表示止损位；SELL 时表示风险反弹位；"
            "HOLD 时表示下方风险位。单位 CNY。"
        ),
    )

    expected_gain_pct: float = Field(
        description=(
            "预期空间百分比。BUY 时通常为正；SELL 时可以表示下方观察空间；"
            "HOLD 时表示上方压力空间。"
        )
    )

    max_loss_pct: float = Field(
        description=(
            "风险空间百分比。BUY 时通常为负数；SELL 时可表示反弹风险；"
            "HOLD 时表示下方风险空间。"
        )
    )

    position_size: float = Field(
        ge=0,
        le=1,
        description="建议仓位，范围 0-1，0 表示空仓，1 表示满仓",
    )

    holding_period: Literal["short_term", "medium_term", "long_term"] = Field(
        description=(
            "建议持有周期："
            "short_term 短线1-5天，medium_term 中线1-4周，long_term 长线1个月以上"
        )
    )

    risk_warning: str = Field(
        description="风险提示，一句话或一段话概括主要风险"
    )

    @field_validator("entry_price_range", mode="before")
    @classmethod
    def normalize_entry_price_range(cls, value):
        """
        防止模型输出 entry_price_range 格式不稳定。

        目标格式：
        {
            "low": 95.5,
            "high": 98.0
        }
        """
        if value is None:
            return {"low": 0.0, "high": 0.0}

        if isinstance(value, dict):
            low = value.get("low", 0.0)
            high = value.get("high", 0.0)

            try:
                low = float(low)
            except (TypeError, ValueError):
                low = 0.0

            try:
                high = float(high)
            except (TypeError, ValueError):
                high = 0.0

            return {"low": low, "high": high}

        return {"low": 0.0, "high": 0.0}

    def _action_emoji(self) -> str:
        return {
            "BUY": "🚀",
            "SELL": "📉",
            "HOLD": "⏸️",
        }.get(self.action, "📌")

    def _action_text(self) -> str:
        if is_english():
            return {
                "BUY": "Buy / Add",
                "SELL": "Sell / Reduce",
                "HOLD": "Hold / Watch",
            }.get(self.action, self.action)

        return {
            "BUY": "买入/加仓",
            "SELL": "卖出/减仓",
            "HOLD": "持有/观望",
        }.get(self.action, self.action)

    def _holding_period_text(self) -> str:
        return {
            "short_term": "Short Term",
            "medium_term": "Medium Term",
            "long_term": "Long Term",
        }.get(self.holding_period, self.holding_period)

    def _trade_plan_labels(self) -> dict[str, str]:
        """
        根据 BUY / SELL / HOLD 动态调整交易计划里的文案。

        这样可以避免 SELL 时仍然显示“进场区间”“止盈目标”“止损防守”。
        """
        if is_english():
            if self.action == "BUY":
                return {
                    "range_label": "Suggested entry range",
                    "target_label": "Profit target",
                    "stop_label": "Stop loss",
                    "expected_label": "Expected upside",
                    "risk_label": "Maximum downside risk",
                    "position_label": "Suggested position size",
                    "extra_note": (
                        "Note: BUY means the setup has some buy/add conditions, "
                        "but execution should still respect personal risk limits."
                    ),
                }

            if self.action == "SELL":
                return {
                    "range_label": "Suggested reduce/exit range",
                    "target_label": "Lower watch level",
                    "stop_label": "Rebound risk level",
                    "expected_label": "Downside watch space",
                    "risk_label": "Rebound risk space",
                    "position_label": "Suggested remaining position",
                    "extra_note": (
                        "Note: SELL means existing holders may consider reducing "
                        "or exiting; it does not imply short selling."
                    ),
                }

            return {
                "range_label": "Watch range",
                "target_label": "Upper resistance",
                "stop_label": "Lower risk level",
                "expected_label": "Upside watch space",
                "risk_label": "Downside risk space",
                "position_label": "Suggested position size",
                "extra_note": (
                    "Note: HOLD means the signal is not decisive enough for "
                    "aggressive action. Avoid chasing rallies or panic selling."
                ),
            }

        if self.action == "BUY":
            return {
                "range_label": "建议进场区间",
                "target_label": "止盈目标（Target）",
                "stop_label": "止损防守（Stop）",
                "expected_label": "预期收益空间",
                "risk_label": "最大亏损风险",
                "position_label": "建议仓位",
                "extra_note": (
                    "说明：BUY 表示当前具备一定买入或加仓条件，"
                    "但仍应结合个人仓位和风险承受能力执行。"
                ),
            }

        if self.action == "SELL":
            return {
                "range_label": "建议减仓/离场区间",
                "target_label": "下方观察位",
                "stop_label": "风险反弹位",
                "expected_label": "下方观察空间",
                "risk_label": "反弹风险空间",
                "position_label": "建议保留仓位",
                "extra_note": (
                    "说明：SELL 在本系统中表示已持有者考虑卖出或减仓，"
                    "未持有者暂时回避；不表示建议做空。"
                ),
            }

        return {
            "range_label": "观察区间",
            "target_label": "上方压力位",
            "stop_label": "下方风险位",
            "expected_label": "上方观察空间",
            "risk_label": "下方风险空间",
            "position_label": "建议仓位",
            "extra_note": (
                "说明：HOLD 表示当前信号不够明确，适合继续观察，"
                "不建议盲目追涨或恐慌卖出。"
            ),
        }

    def to_markdown(
        self,
        symbol: str,
        current_price: float,
        profile_data: str,
        report_date: str | None = None,
    ) -> str:
        """
        将结构化决策转换成命令行展示用 Markdown。

        这个格式对应终端里的 INVESTMENT MEMO。

        修复点：
        1. BUY / SELL / HOLD 使用不同的交易文案。
        2. SELL 不再显示“进场区间”，而显示“建议减仓/离场区间”。
        3. SELL 不再显示“止盈目标/止损防守”，而显示“下方观察位/风险反弹位”。
        """
        report_date = report_date or datetime.now().strftime("%Y-%m-%d")

        low_price = self.entry_price_range.get("low", 0.0)
        high_price = self.entry_price_range.get("high", 0.0)

        emoji = self._action_emoji()
        action_text = self._action_text()
        period_text = self._holding_period_text()
        labels = self._trade_plan_labels()

        if is_english():
            return f"""
> {emoji} **Trading Action: {self.action} / {action_text} (Confidence: {self.confidence:.0%})**

---

## 1. Core Logic (The Why)

{self.core_logic}

---

## 2. Main Force Analysis

{self.main_force_analysis}

---

## 3. Trading Plan (The What)

- Current price: {current_price:.2f} CNY
- {labels["range_label"]}: {low_price:.2f} - {high_price:.2f} CNY
- {labels["target_label"]}: {self.target_price:.2f} CNY ({labels["expected_label"]}: {self.expected_gain_pct:+.1f}%)
- {labels["stop_label"]}: {self.stop_loss:.2f} CNY ({labels["risk_label"]}: {self.max_loss_pct:+.1f}%)
- {labels["position_label"]}: {self.position_size:.0%}
- Holding period: {period_text}

{labels["extra_note"]}

---

## 4. Profile

{profile_data}

---

## 5. Risk Warning

{self.risk_warning}

---

Date: {report_date}
""".strip()

        return f"""
> {emoji} **交易指令：{self.action} / {action_text}（置信度：{self.confidence:.0%}）**

---

## 1. 核心逻辑（The Why）

{self.core_logic}

---

## 2. 主力资金透视（Main Force）

{self.main_force_analysis}

---

## 3. 实战策略（The What）

- 当前价格：{current_price:.2f} CNY
- {labels["range_label"]}：{low_price:.2f} - {high_price:.2f} CNY
- {labels["target_label"]}：{self.target_price:.2f} CNY（{labels["expected_label"]}：{self.expected_gain_pct:+.1f}%）
- {labels["stop_label"]}：{self.stop_loss:.2f} CNY（{labels["risk_label"]}：{self.max_loss_pct:+.1f}%）
- {labels["position_label"]}：{self.position_size:.0%}
- 持有周期：{period_text}

{labels["extra_note"]}

---

## 4. 标的档案（Profile）

{profile_data}

---

## 5. 风险提示 ⚠️

{self.risk_warning}

---

Date: {report_date}
""".strip()
