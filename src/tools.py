"""
A 股智能投研工具层

这个文件只负责“拿数据”和“生成图表”，不负责让 AI 做最终判断。

数据源策略：
1. AkShare 作为主数据源，覆盖行情、资金流、公司档案、实时行情等。
2. 可选启用 akshare-proxy-patch，提高 AkShare 东方财富相关接口成功率。
3. BaoStock 作为兜底数据源，主要兜底历史 K 线、股票名称、行业信息。
4. 所有工具函数失败时尽量返回可读的短错误，不把长异常塞进最终报告。
"""

from __future__ import annotations
from tavily import TavilyClient
from ta.trend import MACD
from ta.momentum import RSIIndicator
from prophet import Prophet
from langchain_core.tools import tool
import pandas as pd
import mplfinance as mpf
import matplotlib
import akshare as ak

import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


# ============================================================
# 0. 环境变量 + AkShare Proxy Patch
# ============================================================
# 重要：
# akshare_proxy_patch 必须在 import akshare as ak 之前安装。
# 否则 AkShare 内部请求可能已经初始化，patch 无法稳定 hook。
load_dotenv()


def env_bool(name: str, default: bool = False) -> bool:
    """
    从环境变量读取布尔值。
    """
    raw = os.getenv(name)

    if raw is None:
        return default

    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def install_akshare_proxy_patch_if_enabled() -> None:
    """
    可选安装 akshare-proxy-patch。

    .env 示例：

    AKSHARE_PROXY_PATCH_ENABLED=true
    AKSHARE_PROXY_PATCH_TOKEN=your_token_here
    AKSHARE_PROXY_PATCH_HOST=101.201.173.125
    AKSHARE_PROXY_PATCH_RETRY=3

    注意：
    - token 从 .env 获取，不能写死在代码里。
    - 这个 patch 只作为 AkShare 的增强，不替代 BaoStock fallback。
    """
    if not env_bool("AKSHARE_PROXY_PATCH_ENABLED", default=False):
        return

    token = os.getenv("AKSHARE_PROXY_PATCH_TOKEN", "").strip()

    if not token:
        print(
            "[Warning] AKSHARE_PROXY_PATCH_ENABLED=true, "
            "but AKSHARE_PROXY_PATCH_TOKEN is missing."
        )
        return

    host = os.getenv("AKSHARE_PROXY_PATCH_HOST", "101.201.173.125").strip()

    try:
        retry = int(os.getenv("AKSHARE_PROXY_PATCH_RETRY", "3"))
    except ValueError:
        retry = 3

    hook_domains_env = os.getenv(
        "AKSHARE_PROXY_PATCH_HOOK_DOMAINS", "").strip()

    if hook_domains_env:
        hook_domains = [
            item.strip()
            for item in hook_domains_env.split(",")
            if item.strip()
        ]
    else:
        hook_domains = [
            "fund.eastmoney.com",
            "push2.eastmoney.com",
            "push2his.eastmoney.com",
            "emweb.securities.eastmoney.com",
        ]

    try:
        import akshare_proxy_patch

        akshare_proxy_patch.install_patch(
            host,
            auth_token=token,
            retry=retry,
            hook_domains=hook_domains,
        )

        print(
            "[Info] akshare_proxy_patch enabled. "
            f"host={host}, retry={retry}, hook_domains={hook_domains}"
        )

    except Exception as e:
        print(f"[Warning] akshare_proxy_patch failed to load: {e}")


install_akshare_proxy_patch_if_enabled()


# ============================================================
# 1. 第三方库导入
# ============================================================
# 注意：akshare 必须在 proxy patch 安装之后导入。

matplotlib.use("Agg")


try:
    import baostock as bs
except Exception:
    bs = None


warnings.filterwarnings("ignore")


# ============================================================
# 2. 通用辅助函数
# ============================================================
def normalize_symbol(symbol: str) -> str:
    """
    统一股票代码格式。

    支持：
    - 600519
    - sh600519
    - sz000001
    - 600519.SH
    - 000001.SZ

    返回：
    - 600519
    - 000001
    """
    symbol = str(symbol).strip().upper()
    symbol = symbol.replace("SH", "")
    symbol = symbol.replace("SZ", "")
    symbol = symbol.replace("BJ", "")
    symbol = symbol.replace(".", "")
    symbol = symbol.replace(" ", "")
    return symbol


def infer_market(symbol: str) -> str:
    """
    根据 A 股代码推断市场。

    - 6 开头：上海 sh
    - 0 / 3 开头：深圳 sz
    - 8 / 4 开头：北交所 bj
    """
    symbol = normalize_symbol(symbol)

    if symbol.startswith("6"):
        return "sh"

    if symbol.startswith(("0", "3")):
        return "sz"

    if symbol.startswith(("8", "4")):
        return "bj"

    return "sh"


def to_baostock_code(symbol: str) -> str:
    """
    转换成 BaoStock 代码格式。

    BaoStock 需要：
    - sh.600519
    - sz.000001
    """
    symbol = normalize_symbol(symbol)

    if symbol.startswith("6"):
        return f"sh.{symbol}"

    if symbol.startswith(("0", "3")):
        return f"sz.{symbol}"

    if symbol.startswith(("8", "4")):
        return f"bj.{symbol}"

    return symbol


def today_yyyymmdd() -> str:
    return datetime.now().strftime("%Y%m%d")


def days_ago_yyyymmdd(days: int) -> str:
    return (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")


def today_yyyy_mm_dd() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def days_ago_yyyy_mm_dd(days: int) -> str:
    return (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")


def to_numeric_safe(series: pd.Series) -> pd.Series:
    """
    安全转换成数字列。
    """
    return pd.to_numeric(series, errors="coerce")


def short_error(e: Exception) -> str:
    """
    把异常转换成短错误，避免最终报告里出现很长的底层异常。
    """
    name = e.__class__.__name__
    text = str(e).strip().replace("\n", " ")

    if len(text) > 120:
        text = text[:120] + "..."

    return f"{name}: {text}" if text else name


def dataframe_tail_to_text(df: pd.DataFrame, rows: int = 5) -> str:
    """
    把 DataFrame 最后几行转成文本，方便返回给 LLM。
    """
    if df is None or df.empty:
        return "No data available."

    return df.tail(rows).to_string(index=False)


def standardize_price_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    统一行情 DataFrame 字段，并转成标准格式。

    标准字段：
    date, open, close, high, low, volume, amount, pct_change, turnover
    """
    if df is None or df.empty:
        return pd.DataFrame()

    rename_map = {
        "日期": "date",
        "股票代码": "symbol",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_change",
        "涨跌额": "change",
        "换手率": "turnover",
        "pctChg": "pct_change",
        "turn": "turnover",
    }

    df = df.rename(columns=rename_map)

    required_cols = ["date", "open", "close", "high", "low", "volume"]
    for col in required_cols:
        if col not in df.columns:
            print(
                f"[Warning] Missing required price column after standardization: {col}")
            return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    numeric_cols = [
        "open",
        "close",
        "high",
        "low",
        "volume",
        "amount",
        "amplitude",
        "pct_change",
        "change",
        "turnover",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = to_numeric_safe(df[col])

    df = df.dropna(subset=["date", "open", "close", "high", "low"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    给行情数据增加 RSI、MACD、均线等指标。
    """
    if df is None or df.empty:
        return pd.DataFrame()

    if len(df) < 30:
        print(
            f"[Warning] Not enough price data to calculate full indicators. Rows: {len(df)}")
        return df

    try:
        df["rsi"] = RSIIndicator(close=df["close"], window=14).rsi()

        macd_indicator = MACD(
            close=df["close"],
            window_slow=26,
            window_fast=12,
            window_sign=9,
        )

        df["macd"] = macd_indicator.macd()
        df["macd_signal"] = macd_indicator.macd_signal()
        df["macd_diff"] = macd_indicator.macd_diff()

        df["ma5"] = df["close"].rolling(window=5).mean()
        df["ma10"] = df["close"].rolling(window=10).mean()
        df["ma20"] = df["close"].rolling(window=20).mean()
        df["ma60"] = df["close"].rolling(window=60).mean()

    except Exception as e:
        print(
            f"[Warning] Technical indicator calculation failed: {short_error(e)}")

    return df


# ============================================================
# 3. AkShare / BaoStock 行情数据
# ============================================================
def fetch_stock_data_akshare(symbol: str, days: int = 180) -> pd.DataFrame:
    """
    使用 AkShare 获取 A 股历史行情。
    如果启用了 akshare-proxy-patch，AkShare 内部请求会自动走 patch。
    """
    symbol = normalize_symbol(symbol)

    try:
        raw_df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=days_ago_yyyymmdd(days),
            end_date=today_yyyymmdd(),
            adjust="qfq",
        )

    except Exception as e:
        print(
            f"[Warning] AkShare stock_zh_a_hist failed for {symbol}: {short_error(e)}")
        return pd.DataFrame()

    if raw_df is None or raw_df.empty:
        print(
            f"[Warning] AkShare stock_zh_a_hist returned empty data for {symbol}")
        return pd.DataFrame()

    return standardize_price_dataframe(raw_df)


def fetch_stock_data_baostock(symbol: str, days: int = 180) -> pd.DataFrame:
    """
    使用 BaoStock 获取历史 K 线，作为 AkShare 的兜底。
    """
    symbol = normalize_symbol(symbol)

    if bs is None:
        print("[Warning] BaoStock is not installed. Run: uv add baostock")
        return pd.DataFrame()

    bs_code = to_baostock_code(symbol)

    login_result = bs.login()
    if login_result.error_code != "0":
        print(f"[Warning] BaoStock login failed: {login_result.error_msg}")
        return pd.DataFrame()

    try:
        rs = bs.query_history_k_data_plus(
            bs_code,
            "date,code,open,high,low,close,volume,amount,pctChg,turn",
            start_date=days_ago_yyyy_mm_dd(days),
            end_date=today_yyyy_mm_dd(),
            frequency="d",
            adjustflag="2",
        )

        if rs.error_code != "0":
            print(f"[Warning] BaoStock history query failed: {rs.error_msg}")
            return pd.DataFrame()

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())

        if not rows:
            print(
                f"[Warning] BaoStock returned empty history data for {symbol}")
            return pd.DataFrame()

        raw_df = pd.DataFrame(rows, columns=rs.fields)
        return standardize_price_dataframe(raw_df)

    except Exception as e:
        print(
            f"[Warning] BaoStock history fallback failed for {symbol}: {short_error(e)}")
        return pd.DataFrame()

    finally:
        try:
            bs.logout()
        except Exception:
            pass


@tool
def get_stock_data(symbol: str, days: int = 180) -> pd.DataFrame:
    """
    获取 A 股历史行情数据，并计算 RSI、MACD 等技术指标。

    数据源顺序：
    1. AkShare stock_zh_a_hist
    2. BaoStock query_history_k_data_plus 兜底

    Args:
        symbol: A 股股票代码，例如 600519、000001、300750
        days: 获取最近多少天数据，默认 180 天

    Returns:
        pandas DataFrame，包含标准化字段：
        date, open, close, high, low, volume, amount, pct_change, turnover,
        rsi, macd, macd_signal, macd_diff
    """
    symbol = normalize_symbol(symbol)

    df = fetch_stock_data_akshare(symbol=symbol, days=days)

    if df is None or df.empty:
        print(f"[Info] Falling back to BaoStock history data for {symbol}...")
        df = fetch_stock_data_baostock(symbol=symbol, days=days)

    if df is None or df.empty:
        print(f"[Warning] All history data sources failed for {symbol}.")
        return pd.DataFrame()

    df = add_technical_indicators(df)
    return df


# ============================================================
# 4. 生成 K 线图
# ============================================================
@tool
def generate_stock_chart(df: Any, symbol: str) -> str:
    """
    根据行情 DataFrame 生成 K 线图。

    Args:
        df: get_stock_data 返回的 DataFrame
        symbol: 股票代码

    Returns:
        生成的图片路径，例如 temp_charts/600519_chart.png
    """
    symbol = normalize_symbol(symbol)

    if df is None or getattr(df, "empty", True):
        return ""

    chart_dir = Path("temp_charts")
    chart_dir.mkdir(parents=True, exist_ok=True)

    chart_path = chart_dir / f"{symbol}_chart.png"

    try:
        plot_df = df.copy()
        plot_df["date"] = pd.to_datetime(plot_df["date"], errors="coerce")
        plot_df = plot_df.dropna(subset=["date"])
        plot_df = plot_df.set_index("date")

        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in plot_df.columns:
                print(f"[Warning] Missing chart column: {col}")
                return ""

        plot_df = plot_df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )

        plot_df = plot_df.tail(90)

        add_plots = []

        if "macd" in df.columns and "macd_signal" in df.columns:
            macd_df = df.copy()
            macd_df["date"] = pd.to_datetime(macd_df["date"], errors="coerce")
            macd_df = macd_df.dropna(subset=["date"]).set_index("date")
            macd_df = macd_df.tail(90)

            add_plots.append(
                mpf.make_addplot(
                    macd_df["macd"],
                    panel=2,
                    ylabel="MACD",
                    width=1,
                )
            )
            add_plots.append(
                mpf.make_addplot(
                    macd_df["macd_signal"],
                    panel=2,
                    width=1,
                )
            )

            if "macd_diff" in macd_df.columns:
                add_plots.append(
                    mpf.make_addplot(
                        macd_df["macd_diff"],
                        type="bar",
                        panel=2,
                        alpha=0.5,
                    )
                )

        mpf.plot(
            plot_df,
            type="candle",
            style="yahoo",
            volume=True,
            mav=(5, 10, 20),
            addplot=add_plots if add_plots else None,
            title=f"{symbol} A-Share Candlestick Chart",
            ylabel="Price",
            ylabel_lower="Volume",
            figsize=(14, 9),
            panel_ratios=(3, 1, 1) if add_plots else (3, 1),
            savefig=dict(fname=str(chart_path), dpi=140, bbox_inches="tight"),
        )

        print(f"Chart saved at: {chart_path}")
        return str(chart_path)

    except Exception as e:
        print(
            f"[Warning] generate_stock_chart failed for {symbol}: {short_error(e)}")
        return ""


# ============================================================
# 5. Prophet 时间序列预测
# ============================================================
@tool
def forecast_with_prophet(df: Any, periods: int = 5) -> str:
    """
    使用 Prophet 对未来几个交易日做短期趋势预测。

    Args:
        df: get_stock_data 返回的 DataFrame
        periods: 预测未来多少天，默认 5 天

    Returns:
        文本形式的预测摘要
    """
    if df is None or getattr(df, "empty", True):
        return "Prophet forecast failed: no price data."

    try:
        prophet_df = df[["date", "close"]].copy()
        prophet_df = prophet_df.rename(columns={"date": "ds", "close": "y"})
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"], errors="coerce")
        prophet_df["y"] = pd.to_numeric(prophet_df["y"], errors="coerce")
        prophet_df = prophet_df.dropna(subset=["ds", "y"])

        if len(prophet_df) < 60:
            return f"Prophet forecast skipped: not enough data. Rows={len(prophet_df)}"

        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05,
        )

        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)

        last_close = float(prophet_df.iloc[-1]["y"])

        future_forecast = forecast.tail(periods)[
            ["ds", "yhat", "yhat_lower", "yhat_upper"]
        ].copy()

        future_forecast["ds"] = future_forecast["ds"].dt.strftime("%Y-%m-%d")

        final_yhat = float(future_forecast.iloc[-1]["yhat"])
        expected_pct = (final_yhat - last_close) / last_close * 100

        if expected_pct > 3:
            trend = "短期预测偏强"
        elif expected_pct < -3:
            trend = "短期预测偏弱"
        else:
            trend = "短期预测震荡"

        forecast_text = future_forecast.to_string(index=False)

        return (
            f"{trend}。\n"
            f"最新收盘价: {last_close:.2f}\n"
            f"{periods}天后预测价: {final_yhat:.2f}\n"
            f"预测涨跌幅: {expected_pct:+.2f}%\n\n"
            f"预测明细:\n{forecast_text}"
        )

    except Exception as e:
        print(f"[Warning] forecast_with_prophet failed: {short_error(e)}")
        return f"Prophet forecast failed: {short_error(e)}"


# ============================================================
# 6. 公司档案：AkShare + BaoStock 兜底
# ============================================================
def fetch_stock_profile_akshare_info(symbol: str) -> str:
    """
    使用 AkShare stock_individual_info_em 获取公司档案。
    如果启用了 akshare-proxy-patch，AkShare 内部请求会自动走 patch。
    """
    symbol = normalize_symbol(symbol)

    try:
        info_df = ak.stock_individual_info_em(symbol=symbol)
    except Exception as e:
        print(
            f"[Warning] AkShare stock_individual_info_em failed for {symbol}: {short_error(e)}")
        return ""

    if info_df is None or info_df.empty:
        return ""

    if "item" in info_df.columns and "value" in info_df.columns:
        info = dict(zip(info_df["item"], info_df["value"]))
    elif "项目" in info_df.columns and "值" in info_df.columns:
        info = dict(zip(info_df["项目"], info_df["值"]))
    else:
        info = {}

    if not info:
        return ""

    name = (
        info.get("股票简称")
        or info.get("简称")
        or info.get("名称")
        or info.get("股票名称")
        or "N/A"
    )

    total_mv = (
        info.get("总市值")
        or info.get("流通市值")
        or info.get("总股本")
        or "N/A"
    )

    industry = (
        info.get("行业")
        or info.get("所属行业")
        or info.get("板块")
        or "N/A"
    )

    listing_date = (
        info.get("上市时间")
        or info.get("上市日期")
        or "N/A"
    )

    lines = [
        f"Symbol: {symbol}",
        "Profile Source: AkShare stock_individual_info_em",
        f"Name: {name}",
        f"Sector: {industry}",
        f"Market Cap: {total_mv}",
        f"Listing Date: {listing_date}",
    ]

    other_items = []
    skip_keys = {
        "股票简称",
        "简称",
        "名称",
        "股票名称",
        "总市值",
        "流通市值",
        "总股本",
        "行业",
        "所属行业",
        "板块",
        "上市时间",
        "上市日期",
    }

    for key, value in info.items():
        key_str = str(key)
        if key_str in skip_keys:
            continue

        other_items.append(f"{key_str}: {value}")

    if other_items:
        lines.append("\nRaw Profile Items:")
        lines.extend(other_items[:8])

    return "\n".join(lines)


def fetch_stock_profile_akshare_spot(symbol: str) -> str:
    """
    使用 AkShare stock_zh_a_spot_em 兜底获取名称、价格、市值等。
    如果启用了 akshare-proxy-patch，AkShare 内部请求会自动走 patch。
    """
    symbol = normalize_symbol(symbol)

    try:
        spot_df = ak.stock_zh_a_spot_em()
    except Exception as e:
        print(
            f"[Warning] AkShare stock_zh_a_spot_em failed for {symbol}: {short_error(e)}")
        return ""

    if spot_df is None or spot_df.empty or "代码" not in spot_df.columns:
        return ""

    row_df = spot_df[spot_df["代码"].astype(str) == symbol]

    if row_df.empty:
        return ""

    row = row_df.iloc[0]

    name = row.get("名称", "N/A")
    latest_price = row.get("最新价", "N/A")
    pct_change = row.get("涨跌幅", "N/A")
    total_mv = row.get("总市值", "N/A")
    turnover = row.get("换手率", "N/A")

    return "\n".join(
        [
            f"Symbol: {symbol}",
            "Profile Source: AkShare stock_zh_a_spot_em fallback",
            f"Name: {name}",
            f"Latest Price: {latest_price}",
            f"Pct Change: {pct_change}",
            f"Market Cap: {total_mv}",
            f"Turnover: {turnover}",
        ]
    )


def fetch_stock_profile_baostock(symbol: str) -> str:
    """
    使用 BaoStock 获取股票基础信息和行业信息，作为公司档案兜底。
    """
    symbol = normalize_symbol(symbol)

    if bs is None:
        print("[Warning] BaoStock is not installed. Run: uv add baostock")
        return ""

    bs_code = to_baostock_code(symbol)

    login_result = bs.login()
    if login_result.error_code != "0":
        print(f"[Warning] BaoStock login failed: {login_result.error_msg}")
        return ""

    lines = [
        f"Symbol: {symbol}",
        "Profile Source: BaoStock fallback",
    ]

    try:
        basic_rs = bs.query_stock_basic(code=bs_code)

        if basic_rs.error_code == "0":
            basic_rows = []
            while basic_rs.next():
                basic_rows.append(basic_rs.get_row_data())

            if basic_rows:
                basic_df = pd.DataFrame(basic_rows, columns=basic_rs.fields)
                row = basic_df.iloc[0].to_dict()

                lines.extend(
                    [
                        f"Name: {row.get('code_name', 'N/A')}",
                        f"Code: {row.get('code', bs_code)}",
                        f"IPO Date: {row.get('ipoDate', 'N/A')}",
                        f"Out Date: {row.get('outDate', 'N/A')}",
                        f"Type: {row.get('type', 'N/A')}",
                        f"Status: {row.get('status', 'N/A')}",
                    ]
                )

        industry_rs = bs.query_stock_industry(code=bs_code)

        if industry_rs.error_code == "0":
            industry_rows = []
            while industry_rs.next():
                industry_rows.append(industry_rs.get_row_data())

            if industry_rows:
                industry_df = pd.DataFrame(
                    industry_rows, columns=industry_rs.fields)
                row = industry_df.iloc[0].to_dict()

                lines.extend(
                    [
                        f"Industry: {row.get('industry', 'N/A')}",
                        f"Industry Classification: {row.get('industryClassification', 'N/A')}",
                    ]
                )

    except Exception as e:
        print(
            f"[Warning] BaoStock profile fallback failed for {symbol}: {short_error(e)}")
        return ""

    finally:
        try:
            bs.logout()
        except Exception:
            pass

    if len(lines) <= 2:
        return ""

    return "\n".join(lines)


def fetch_stock_name_akshare(symbol: str) -> str:
    """
    使用 AkShare 股票代码名称表做最后兜底。
    如果启用了 akshare-proxy-patch，AkShare 内部请求会自动走 patch。
    """
    symbol = normalize_symbol(symbol)

    try:
        code_name_df = ak.stock_info_a_code_name()
    except Exception as e:
        print(
            f"[Warning] AkShare stock_info_a_code_name failed for {symbol}: {short_error(e)}")
        return ""

    if code_name_df is None or code_name_df.empty:
        return ""

    code_col = (
        "code"
        if "code" in code_name_df.columns
        else "代码"
        if "代码" in code_name_df.columns
        else None
    )

    name_col = (
        "name"
        if "name" in code_name_df.columns
        else "名称"
        if "名称" in code_name_df.columns
        else None
    )

    if not code_col or not name_col:
        return ""

    row_df = code_name_df[code_name_df[code_col].astype(str) == symbol]

    if row_df.empty:
        return ""

    name = row_df.iloc[0][name_col]

    return "\n".join(
        [
            f"Symbol: {symbol}",
            "Profile Source: AkShare stock_info_a_code_name fallback",
            f"Name: {name}",
        ]
    )


@tool
def get_stock_profile(symbol: str) -> str:
    """
    获取 A 股公司档案信息。

    数据源顺序：
    1. AkShare stock_individual_info_em
    2. AkShare stock_zh_a_spot_em
    3. BaoStock query_stock_basic + query_stock_industry
    4. AkShare stock_info_a_code_name

    Args:
        symbol: A 股股票代码

    Returns:
        公司档案文本，例如名称、行业、市值、上市时间等
    """
    symbol = normalize_symbol(symbol)

    fetchers = [fetch_stock_profile_baostock]
    if env_bool("ENABLE_AKSHARE_PROFILE_FALLBACK", default=False):
        fetchers = [
            fetch_stock_profile_akshare_info,
            fetch_stock_profile_akshare_spot,
            fetch_stock_profile_baostock,
            fetch_stock_name_akshare,
        ]

    for fetcher in fetchers:
        try:
            profile = fetcher(symbol)

            if profile and "Name:" in profile:
                return profile

        except Exception as e:
            print(
                f"[Warning] Profile fetcher {fetcher.__name__} failed: {short_error(e)}")

    return "\n".join(
        [
            f"Symbol: {symbol}",
            "Company profile fetch failed. All profile data sources are temporarily unavailable.",
        ]
    )


# ============================================================
# 7. Tavily 新闻搜索
# ============================================================
@tool
def search_news(symbol: str, max_results: int = 5) -> str:
    """
    使用 Tavily 搜索该股票的相关新闻、公告、利好利空。

    重点改进：
    1. 搜索 query 明确要求“近期”“公司公告”“财报”“减持”“监管”等关键词。
    2. 返回结果保留 URL、标题、摘要、发布时间。
    3. 如果 Tavily 没有返回发布时间，明确标记 Published Date: Unknown。
    4. 给 News Agent 明确提示：没有日期的新闻不能当作强近期证据。
    5. 避免把完整 profile 错误信息塞进搜索 query。
    """
    symbol = normalize_symbol(symbol)

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "News search failed: TAVILY_API_KEY is missing."

    current_date = datetime.now().strftime("%Y-%m-%d")

    try:
        profile_text = get_stock_profile.invoke({"symbol": symbol})
    except Exception:
        profile_text = f"Symbol: {symbol}"

    company_name = ""
    for line in str(profile_text).splitlines():
        line = line.strip()

        if line.startswith("Name:"):
            company_name = line.replace("Name:", "").strip()
            break

    if company_name and company_name != "N/A":
        search_target = f"{symbol} {company_name}"
    else:
        search_target = symbol

    query = (
        f"{search_target} A股 近30天 最新 新闻 公告 财报 业绩 预告 "
        f"减持 增持 回购 监管处罚 行业政策 利好 利空 风险"
    )

    try:
        client = TavilyClient(api_key=api_key)

        response = client.search(
            query=query,
            search_depth=os.getenv("TAVILY_SEARCH_DEPTH", "basic").strip() or "basic",
            max_results=max_results,
            include_answer=True,
            include_raw_content=False,
        )

        answer = response.get("answer", "")
        results = response.get("results", [])

        if not results and not answer:
            return (
                f"No recent news found for {symbol}.\n"
                f"Current Date: {current_date}\n"
                "News Source Quality: weak\n"
                "Reason: Tavily returned no answer and no search results."
            )

        lines = [
            f"Current Date: {current_date}",
            f"Target Symbol: {symbol}",
            f"Target Company: {company_name or 'Unknown'}",
            f"News search query: {query}",
            "",
            "Important instructions for News Agent:",
            "- Treat items with explicit recent dates as stronger evidence.",
            "- Treat items with Published Date: Unknown as weak evidence.",
            "- Do not present industry-level news as company-specific fact.",
            "- Do not claim a financial result, penalty, reduction, or policy impact unless it is directly supported by a search result.",
            "- If evidence is weak, say so in risk_warnings.",
        ]

        if answer:
            lines.append("\nTavily Answer:")
            lines.append(str(answer))

        if results:
            lines.append("\nSearch Results:")

            for idx, item in enumerate(results, start=1):
                title = item.get("title", "N/A")
                url = item.get("url", "N/A")
                content = item.get("content", "N/A")
                score = item.get("score", "N/A")

                published_date = (
                    item.get("published_date")
                    or item.get("date")
                    or item.get("publishedDate")
                    or item.get("pub_date")
                    or "Unknown"
                )

                title_content = f"{title} {content}"

                if company_name and company_name in title_content:
                    relevance = "direct_company_related"
                elif symbol in title_content:
                    relevance = "direct_symbol_related"
                else:
                    relevance = "possibly_industry_or_market_related"

                lines.append(
                    f"\n[{idx}] {title}\n"
                    f"URL: {url}\n"
                    f"Published Date: {published_date}\n"
                    f"Relevance: {relevance}\n"
                    f"Score: {score}\n"
                    f"Content: {content}"
                )

        return "\n".join(lines)

    except Exception as e:
        print(f"[Warning] Tavily search failed for {symbol}: {short_error(e)}")
        return (
            f"News search failed: {short_error(e)}\n"
            f"Current Date: {current_date}\n"
            "News Source Quality: weak\n"
            "Reason: Tavily API call failed."
        )


# ============================================================
# 8. 获取资金流数据
# ============================================================
@tool
def get_money_flow(symbol: str) -> str:
    """
    获取个股资金流数据。

    注意：
    BaoStock 不适合作为资金流兜底，所以资金流仍然优先使用 AkShare。
    如果启用了 akshare-proxy-patch，AkShare 内部请求会自动走 patch。
    如果失败，返回短错误提示，不影响主流程。

    Args:
        symbol: A 股股票代码

    Returns:
        资金流向摘要文本
    """
    symbol = normalize_symbol(symbol)

    if not env_bool("ENABLE_MONEY_FLOW", default=False):
        return "Money flow disabled by ENABLE_MONEY_FLOW=false."

    market = infer_market(symbol)

    if market == "bj":
        return "Money flow fetch failed: 北交所股票暂未适配该资金流接口。"

    try:
        df = ak.stock_individual_fund_flow(stock=symbol, market=market)

    except Exception as e:
        print(
            f"[Warning] get_money_flow failed for {symbol}: {short_error(e)}")
        return f"Money flow fetch failed: data source unavailable ({short_error(e)})."

    if df is None or df.empty:
        return f"Money flow fetch failed: empty result for {symbol}."

    try:
        recent = df.tail(5).copy()

        numeric_keywords = [
            "净额",
            "净占比",
            "收盘价",
            "涨跌幅",
        ]

        for col in recent.columns:
            if any(keyword in str(col) for keyword in numeric_keywords):
                recent[col] = pd.to_numeric(recent[col], errors="coerce")

        summary_lines = [
            f"Symbol: {symbol}",
            f"Market: {market}",
            "Recent money flow data:",
            recent.to_string(index=False),
        ]

        main_force_cols = [
            col for col in recent.columns if "主力净流入" in str(col)
        ]

        if main_force_cols:
            summary_lines.append("\nMain force related columns:")
            summary_lines.append(
                recent[main_force_cols].to_string(index=False))

        return "\n".join(summary_lines)

    except Exception as e:
        print(
            f"[Warning] money flow parse failed for {symbol}: {short_error(e)}")
        return f"Money flow parse failed: {short_error(e)}"
