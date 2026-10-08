"""
A-Stock AI Investment Assistant CLI

这个文件是项目入口，负责：
1. 加载 .env 环境变量
2. 展示命令行欢迎界面
3. 接收用户输入的 A 股股票代码
4. 调用 LangGraph 工作流 app
5. 将 FinalDecision 输出为 Rich Markdown 投资备忘录
"""

from __future__ import annotations
from src.models import FinalDecision
from src.graph import app

import asyncio
import os
import sys
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from src.settings import is_english

# 确保可以从项目根目录导入 src 包
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


console = Console()


def text_by_lang(cn: str, en: str) -> str:
    """
    Return localized user-facing text.
    """
    return en if is_english() else cn


def normalize_symbol(raw_symbol: str) -> str:
    """
    标准化用户输入的股票代码。

    支持这些输入：
    - 600519
    - 000001
    - 300750
    - sh600519
    - sz000001
    - 600519.SH
    - 000001.SZ

    最终统一转成：
    - 600519
    - 000001
    - 300750
    """
    symbol = raw_symbol.strip().upper()

    symbol = symbol.replace("SH", "")
    symbol = symbol.replace("SZ", "")
    symbol = symbol.replace(".", "")
    symbol = symbol.replace(" ", "")

    return symbol


def is_exit_command(text: str) -> bool:
    """
    判断用户是否想退出程序。
    """
    return text.strip().lower() in {"q", "quit", "exit"}


def print_welcome() -> None:
    """
    打印欢迎界面。
    """
    welcome_text = Text(
        "Multi-Agent Stock Research System",
        justify="center",
        style="bold cyan",
    )

    sub_text = Text(
        "7 Agents: Quant • Vision • News • Bull • Bear • Risk • CIO",
        justify="center",
        style="yellow",
    )

    console.print(
        Panel(
            Text.assemble(welcome_text, "\n", sub_text),
            title="🤖 Multi-Agent Stock Research System",
            border_style="green",
            expand=False,
            padding=(1, 2),
        )
    )

    console.print(
        "[dim](Enter 'q', 'quit' or 'exit' to stop)[/dim]\n",
        justify="center",
    )


def check_env_keys() -> bool:
    """
    检查必要 API Key。

    OPENAI_API_KEY 是核心推理模型必须要的。
    ZHIPUAI_API_KEY 和 TAVILY_API_KEY 不是绝对必须，但缺失会影响 Visual Agent / News Agent。
    """
    ok = True

    if not os.getenv("OPENAI_API_KEY"):
        console.print(
            "[bold red][Error][/bold red] OPENAI_API_KEY is missing. "
            "Please check your .env file."
        )
        ok = False

    if not os.getenv("OPENAI_BASE_URL"):
        console.print(
            "[bold yellow][Warning][/bold yellow] OPENAI_BASE_URL is missing. "
            "If you use OpenRouter, please set OPENAI_BASE_URL=https://openrouter.ai/api/v1"
        )

    if not os.getenv("OPENAI_MODEL_NAME"):
        console.print(
            "[bold yellow][Warning][/bold yellow] OPENAI_MODEL_NAME is missing. "
            "The project may fall back to the default model."
        )

    if not os.getenv("TAVILY_API_KEY"):
        console.print(
            "[bold yellow][Warning][/bold yellow] TAVILY_API_KEY is missing. "
            "News Agent may fail to search latest news."
        )

    return ok


def get_current_price(result: dict[str, Any]) -> float:
    """
    从 LangGraph 返回结果中提取当前价格。

    graph.py 里 fetcher_node 会把 pandas DataFrame 放到 price_data。
    最后一行 close 就是最新收盘价。
    """
    price_data = result.get("price_data")

    if price_data is None:
        return 0.0

    try:
        if hasattr(price_data, "empty") and not price_data.empty:
            return float(price_data.iloc[-1]["close"])
    except Exception:
        return 0.0

    return 0.0


def render_final_report(
    symbol: str,
    result: dict[str, Any],
) -> None:
    """
    渲染最终投资备忘录。
    """
    final_report = result.get("final_report")
    profile_data = result.get("profile_data", "N/A")
    current_price = get_current_price(result)
    report_date = datetime.now().strftime("%Y-%m-%d")

    if isinstance(final_report, FinalDecision):
        markdown_report = final_report.to_markdown(
            symbol=symbol,
            current_price=current_price,
            profile_data=profile_data,
            report_date=report_date,
        )

        console.print(
            Panel(
                Markdown(markdown_report),
                title=f"💰 INVESTMENT MEMO: {symbol}",
                subtitle=f"Date: {report_date}",
                border_style="cyan",
                padding=(1, 2),
                expand=True,
            )
        )
        return

    if final_report is not None:
        console.print(
            Panel(
                str(final_report),
                title=f"📊 Final Report: {symbol}",
                subtitle=f"Date: {report_date}",
                border_style="green",
                padding=(1, 2),
            )
        )
        return

    console.print(
        Panel(
            text_by_lang(
                "[bold red]No final_report generated.[/bold red]\n\n"
                "可能原因：\n"
                "1. CIO 节点没有正常返回结果\n"
                "2. LLM 结构化输出失败\n"
                "3. graph.py 中 final_report 字段名不一致\n"
                "4. models.py 的 FinalDecision 字段和 CIO_PROMPT 不匹配",
                "[bold red]No final_report generated.[/bold red]\n\n"
                "Possible causes:\n"
                "1. The CIO node did not return a valid result\n"
                "2. LLM structured output failed\n"
                "3. final_report field name is inconsistent in graph.py\n"
                "4. FinalDecision fields and CIO_PROMPT are inconsistent",
            ),
            title="❌ Report Error",
            border_style="red",
        )
    )


async def analyze_symbol(symbol: str) -> None:
    """
    分析单只股票。
    """
    console.print(
        Panel(
            text_by_lang(
                f"正在分析股票代码：[bold yellow]{symbol}[/bold yellow]\n\n"
                "系统将执行以下步骤：\n"
                "1. Fetcher 抓取行情、资金流、新闻和公司档案\n"
                "2. Quant Agent 分析量化指标和资金意图\n"
                "3. Visual Agent 查看 K 线图\n"
                "4. News Agent 分析舆情和新闻\n"
                "5. Bull / Bear / Risk Agents 进行牛熊反驳和风控裁决\n"
                "6. CIO Agent 汇总并生成最终交易指令",
                f"Analyzing stock code: [bold yellow]{symbol}[/bold yellow]\n\n"
                "The system will run these steps:\n"
                "1. Fetcher retrieves market data, money flow, news, and company profile\n"
                "2. Quant Agent analyzes quantitative indicators and capital flow intent\n"
                "3. Visual Agent reviews the candlestick chart\n"
                "4. News Agent analyzes news and sentiment\n"
                "5. Bull / Bear / Risk Agents debate the case and perform risk control\n"
                "6. CIO Agent synthesizes everything into a final trading decision",
            ),
            title="🚀 Start Analysis",
            border_style="blue",
            padding=(1, 2),
        )
    )

    console.print(
        f"✨ [bold green]Agents are analyzing {symbol}...[/bold green]")

    try:
        result = await app.ainvoke({"symbol": symbol})
        render_final_report(symbol=symbol, result=result)

        chart_path = result.get("chart_path")
        if chart_path:
            console.print(f"\n🖼️ Chart saved at: [dim]{chart_path}[/dim]")

    except KeyboardInterrupt:
        console.print(
            text_by_lang(
                "\n[bold yellow]用户中断分析。[/bold yellow]",
                "\n[bold yellow]Analysis interrupted by user.[/bold yellow]",
            )
        )
        raise

    except Exception as e:
        console.print(
            Panel(
                text_by_lang(
                    f"[bold red]运行失败：[/bold red]\n\n{e}\n\n"
                    "建议检查：\n"
                    "1. .env 里的 API Key 是否正确\n"
                    "2. 股票代码是否有效\n"
                    "3. akshare 是否能正常获取数据\n"
                    "4. graph.py / agents.py / models.py 字段是否一致",
                    f"[bold red]Runtime failed:[/bold red]\n\n{e}\n\n"
                    "Suggested checks:\n"
                    "1. API keys in .env are correct\n"
                    "2. The stock code is valid\n"
                    "3. akshare can fetch data normally\n"
                    "4. graph.py / agents.py / models.py fields are consistent",
                ),
                title="❌ Runtime Error",
                border_style="red",
                padding=(1, 2),
            )
        )


async def main() -> None:
    """
    CLI 主循环。
    """
    load_dotenv()

    print_welcome()

    if not check_env_keys():
        return

    while True:
        raw_symbol = Prompt.ask(
            text_by_lang(
                "[bold yellow]Enter Stock Code[/bold yellow] [dim](e.g., 600519 or q to quit)[/dim]",
                "[bold yellow]Enter Stock Code[/bold yellow] [dim](e.g., 600519 or q to quit)[/dim]",
            )
        )

        if is_exit_command(raw_symbol):
            console.print("\n👋 [bold green]Exiting. Goodbye![/bold green]")
            break

        symbol = normalize_symbol(raw_symbol)

        if not symbol:
            console.print(
                text_by_lang(
                    "[bold yellow]股票代码不能为空，请重新输入。[/bold yellow]",
                    "[bold yellow]Stock code cannot be empty. Please try again.[/bold yellow]",
                )
            )
            continue

        if not symbol.isdigit() or len(symbol) != 6:
            console.print(
                text_by_lang(
                    "[bold yellow]股票代码格式可能不正确。请输入 6 位 A 股代码，例如 600519、000001、300750。[/bold yellow]",
                    "[bold yellow]The stock code format looks invalid. Please enter a 6-digit A-share code, such as 600519, 000001, or 300750.[/bold yellow]",
                )
            )
            continue

        await analyze_symbol(symbol)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n👋 [bold green]Exiting. Goodbye![/bold green]")
