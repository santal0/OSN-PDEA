"""
Dark Pattern Differential Exposure Workbench
一站式命令行实验工作台：profile 选择/切换、动作生成、截图、DPS 逐维标注、撤回、统计分析、导出结果。

运行：
    python dark_pattern_workbench.py

可选依赖：
    pip install pyautogui pillow matplotlib pandas numpy
其中 pyautogui/pillow 用于自动截图；matplotlib/pandas/numpy 用于更好的导出和统计。
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except Exception:
    np = None

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import pyautogui
except Exception:
    pyautogui = None

DPS_DIMS = ["I", "F", "P", "C", "S", "N", "FA"]

DPS_GUIDE = {
    "I": {
        "name": "Interface Interference / 接口干扰",
        "scale": "0/1/2",
        "rule": "0=无明显诱导；1=轻微突出/弱化选项；2=明显通过颜色、大小、位置、隐藏关闭等强诱导。",
        "examples": "同意按钮巨大、拒绝灰色、关闭按钮极小、重要信息字体过小或折叠。",
    },
    "F": {
        "name": "Friction / 交互摩擦",
        "scale": "整数，建议 0-5",
        "rule": "F = 退出/拒绝/关闭所需操作数 - 接受/继续所需操作数；最低可填 0。",
        "examples": "接受 1 步，拒绝 3 步，则 F=2。",
    },
    "P": {
        "name": "Pressure / 行为压力",
        "scale": "0/1",
        "rule": "是否出现倒计时、稀缺性、社会证明、情绪施压等。0=无，1=有。",
        "examples": "仅剩 2 件、限时 00:59、大家都在买、错过就亏。",
    },
    "C": {
        "name": "Contextual Deception / 上下文欺骗",
        "scale": "0/1",
        "rule": "按钮/文案/图标语义与实际结果是否不一致。0=无，1=有。",
        "examples": "点击 X 却确认同意；关闭弹窗却进入购买页。",
    },
    "S": {
        "name": "Sneaking / Preselection / 预设偷运",
        "scale": "0/1",
        "rule": "是否默认勾选、自动续费、静默加入购物车或隐藏附加服务。0=无，1=有。",
        "examples": "默认勾选会员续费、默认添加保险/礼包。",
    },
    "N": {
        "name": "Nagging / 反复纠缠",
        "scale": "整数，建议 0-5；也可 0/1",
        "rule": "本步骤中被弹窗/提示/评分/升级打断的次数；若只做二值标注，有打断填 1。",
        "examples": "连续弹出评分、开会员、通知权限。",
    },
    "FA": {
        "name": "Forced Action / Restrictive / 强制行为",
        "scale": "0/1",
        "rule": "是否必须注册、授权、分享隐私、下载、付费或执行额外任务才能继续。0=无，1=有。",
        "examples": "不允许跳过登录；必须开通讯录权限才能使用。",
    },
}

DEFAULT_CONFIG = {
    "app": "APP_NAME",
    "max_steps": 30,
    "random_seed": 20260603,
    "screenshot_after_each_action": True,
    "stability_window": 5,
    "stability_epsilon": 0.10,
    "differential_tau": 0.20,
    "screenshot_delay_seconds": 3,
    "output_dir": "results",
}

DEFAULT_KEYWORDS = {
    "shopping": ["会员", "百亿补贴", "限时秒杀", "满减券", "拼单"],
    "daily_necessities": ["纸巾", "洗发水", "洗衣液", "零食", "垃圾袋"],
    "electronics_high_value": ["手机", "联想笔记本", "索尼相机", "平板电脑", "手柄"],
    "apparel_beauty": ["连衣裙", "高跟鞋", "防晒衣", "粉底液", "面膜"],
}
DEFAULT_PROFILES = {
    "high_payment_shopping": {
        "description": "高付费意向/数码发烧友画像。频繁进入核心决策与高客单价结算页，是‘预设分期、默认加购保险、诱导办大额会员’的主要受害者。",
        "behavior": {
            "visit_homepage": 0.05,
            "search_keyword": 0.25,            # 高频主动搜索目标明确的商品
            "browse_or_scroll_page": 0.40,
            "decision_interface_visit": 0.30,  # 高频点击购买、配置参数、进入结算
        },
        "key_words": {
            "electronics_high_value": 0.50,    # 半数搜索集中在高客单价数码，以此高频触发金融、分期、捆绑暗黑模式
            "shopping": 0.20,        # 寻找大额券
            "apparel_beauty": 0.20,            # 常规潮流消费
            "daily_necessities": 0.10,          # 极少在平台浪费时间搜日用品
        },
    },
    "low_payment_browsing": {
        "description": "低付费/闲逛消磨时间画像。偏向首页和信息流的日常浏览，极少点击购买，主要测试信息流中的‘伪装广告、低价诱导点击’等暗黑模式。",
        "behavior": {
            "visit_homepage": 0.15,            # 经常回首页刷推荐流
            "search_keyword": 0.15,            # 较少主动搜索
            "browse_or_scroll_page": 0.60,     # 绝大部分时间在无目的地滑动页面
            "decision_interface_visit": 0.10,  # 极少进入结算或深层决策界面
        },
        "key_words": {
            "apparel_beauty": 0.45,            # 闲逛时最爱看服饰、美妆推荐
            "daily_necessities": 0.35,         # 偶尔搜搜刚需日用品对比价格
            "shopping": 0.15,        # 随便看看有什么秒杀
            "electronics_high_value": 0.05,    # 几乎不看高价大件
        },
    },
    "bargain_hunter": {
        "description": "促销敏感/薅羊毛画像。高频流连于各种低价会场，极易触发‘摇红包弹窗反复打断、砍一刀交互摩擦、默认勾选先用后付’等暗黑模式。",
        "behavior": {
            "visit_homepage": 0.10,
            "search_keyword": 0.35,            # 高频通过搜索栏寻找特定活动和低价商品
            "browse_or_scroll_page": 0.35,
            "decision_interface_visit": 0.20,  # 频繁点击“免费领”、“抽奖”、“领福利”等高诱导性按钮
        },
        "key_words": {
            "shopping": 0.60,
            "daily_necessities": 0.30,         # 倾向于在平台购买便宜的纸巾、零食等刚需
            "apparel_beauty": 0.10,            # 偶尔看看低价服饰
            "electronics_high_value": 0.00,
        },
    },
}

@dataclass
class StepRecord:
    app: str
    profile_id: str
    step: int
    timestamp: str
    action: str
    category: Optional[str]
    keyword: Optional[str]
    instruction: str
    screenshot_path: Optional[str]
    dps: Dict[str, float]
    note: str = ""


def ensure_default_files(base: Path) -> Tuple[Path, Path]:
    base.mkdir(parents=True, exist_ok=True)
    config_path = base / "config.json"
    profiles_path = base / "profiles.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")
    if not profiles_path.exists():
        profiles_path.write_text(json.dumps({"profiles": DEFAULT_PROFILES, "keywords": DEFAULT_KEYWORDS}, ensure_ascii=False, indent=2), encoding="utf-8")
    return config_path, profiles_path


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def normalize_dist(dist: Dict[str, float]) -> Dict[str, float]:
    cleaned = {k: max(0.0, float(v)) for k, v in dist.items()}
    total = sum(cleaned.values())
    if total <= 0:
        raise ValueError(f"概率分布总和必须大于 0: {dist}")
    return {k: v / total for k, v in cleaned.items()}


def weighted_choice(dist: Dict[str, float], rng: random.Random) -> str:
    norm = normalize_dist(dist)
    r = rng.random()
    acc = 0.0
    last = next(iter(norm))
    for k, p in norm.items():
        acc += p
        last = k
        if r <= acc:
            return k
    return last


def make_instruction(app: str, action: str, category: Optional[str], keyword: Optional[str], delay: int) -> str:
    if action == "visit_homepage":
        return f"请回到 {app} 首页，等待 {delay} 秒后截图。"
    if action == "search_keyword":
        return f"请在搜索框输入“{keyword}”，点击第一个自然/推荐结果，等待 {delay} 秒后截图。"
    if action == "browse_or_scroll_page":
        return f"请在当前页面向下浏览/滑动一次，停留观察 {delay} 秒后截图。"
    if action == "decision_interface_visit":
        return f"请点击当前屏幕中最明显的决策入口，例如内容详情、会员、订阅、充值、购买、领取优惠等，等待 {delay} 秒后截图。"
    return f"请执行动作 {action}，等待 {delay} 秒后截图。"


def generate_action(profile: Dict[str, Any], keywords: Dict[str, List[str]], rng: random.Random, app: str, delay: int) -> Tuple[str, Optional[str], Optional[str], str]:
    action = weighted_choice(profile["behavior"], rng)
    category = None
    keyword = None
    if action == "search_keyword":
        category = weighted_choice(profile.get("key_words", {}), rng)
        choices = keywords.get(category, [])
        keyword = rng.choice(choices) if choices else category
    instruction = make_instruction(app, action, category, keyword, delay)
    return action, category, keyword, instruction


def take_screenshot(path: Path, delay: int) -> Optional[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n即将截图：等待 {delay} 秒。请把模拟器/APP窗口置于目标画面。")
    time.sleep(max(0, delay))
    if pyautogui is None:
        print("未检测到 pyautogui，无法自动截图。你可以手动截图后把文件放到该路径：")
        print(path)
        input("手动截图完成后按 Enter 继续；如果没有截图也可直接继续。")
        return str(path) if path.exists() else None
    try:
        img = pyautogui.screenshot()
        img.save(path)
        print(f"截图已保存：{path}")
        return str(path)
    except Exception as e:
        print(f"自动截图失败：{e}")
        input("请手动截图后按 Enter 继续。")
        return str(path) if path.exists() else None


def prompt_float(dim: str) -> float:
    guide = DPS_GUIDE[dim]
    while True:
        print(f"\n[{dim}] {guide['name']}")
        print(f"标尺：{guide['scale']}")
        print(f"判别：{guide['rule']}")
        print(f"例子：{guide['examples']}")
        raw = input(f"请输入 {dim}：").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            raise KeyboardInterrupt
        try:
            val = float(raw)
            if val < 0:
                print("请输入非负数。")
                continue
            return val
        except ValueError:
            print("输入无效，请输入数字，例如 0、1、2。")


def prompt_dps() -> Dict[str, float]:
    print("\n开始逐维填写 DPS=[I,F,P,C,S,N,FA]。输入 q 可中断本步。")
    return {dim: prompt_float(dim) for dim in DPS_DIMS}


def records_to_rows(records: List[StepRecord]) -> List[Dict[str, Any]]:
    rows = []
    for r in records:
        row = asdict(r)
        dps = row.pop("dps")
        for dim in DPS_DIMS:
            row[dim] = dps.get(dim, 0.0)
        rows.append(row)
    return rows


def write_records(records: List[StepRecord], run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = records_to_rows(records)
    jsonl = run_dir / "steps.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    csv_path = run_dir / "steps.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def load_all_records(output_dir: Path) -> List[StepRecord]:
    records: List[StepRecord] = []
    for p in output_dir.glob("**/steps.jsonl"):
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    dps = {dim: float(obj.get(dim, 0)) for dim in DPS_DIMS}
                    records.append(StepRecord(
                        app=obj["app"], profile_id=obj["profile_id"], step=int(obj["step"]),
                        timestamp=obj.get("timestamp", ""), action=obj.get("action", ""),
                        category=obj.get("category"), keyword=obj.get("keyword"),
                        instruction=obj.get("instruction", ""), screenshot_path=obj.get("screenshot_path"),
                        dps=dps, note=obj.get("note", "")
                    ))
        except Exception as e:
            print(f"读取失败 {p}: {e}")
    return records


def mean_ci(values: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = sum(values) / n
    if n == 1:
        return mean, mean, mean
    sd = statistics.stdev(values)
    # n<=30 用 2.0 近似；课程作业足够直观，也避免 scipy 依赖。
    z = 1.96 if n > 30 else 2.0
    half = z * sd / math.sqrt(n)
    return mean, mean - half, mean + half


def summarize(records: List[StepRecord]) -> Dict[str, Any]:
    by_profile: Dict[str, List[StepRecord]] = {}
    for r in records:
        by_profile.setdefault(r.profile_id, []).append(r)
    summary: Dict[str, Any] = {}
    for pid, rs in by_profile.items():
        item = {"n": len(rs), "E": {}, "ci95": {}}
        for dim in DPS_DIMS:
            vals = [float(r.dps.get(dim, 0)) for r in rs]
            m, lo, hi = mean_ci(vals)
            item["E"][dim] = m
            item["ci95"][dim] = [lo, hi]
        summary[pid] = item
    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    if not summary:
        print("暂无可统计记录。")
        return
    print("\n=== 条件暴露率 E(D_i | A) 与 95% CI ===")
    header = "Profile".ljust(26) + "n".rjust(5) + "  " + "  ".join(d.rjust(14) for d in DPS_DIMS)
    print(header)
    print("-" * len(header))
    for pid, item in summary.items():
        parts = [pid[:26].ljust(26), str(item["n"]).rjust(5)]
        for dim in DPS_DIMS:
            m = item["E"][dim]
            lo, hi = item["ci95"][dim]
            parts.append(f"{m:.2f}[{lo:.2f},{hi:.2f}]".rjust(14))
        print("  ".join(parts))


def pairwise_diff(summary: Dict[str, Any], tau: float) -> List[Dict[str, Any]]:
    profiles = list(summary.keys())
    rows = []
    for i in range(len(profiles)):
        for j in range(i + 1, len(profiles)):
            a, b = profiles[i], profiles[j]
            for dim in DPS_DIMS:
                diff = summary[a]["E"][dim] - summary[b]["E"][dim]
                rows.append({
                    "profile_a": a, "profile_b": b, "dimension": dim,
                    "E_a": summary[a]["E"][dim], "E_b": summary[b]["E"][dim],
                    "diff_a_minus_b": diff, "abs_diff": abs(diff),
                    "exceeds_tau": abs(diff) > tau,
                })
    return rows


def export_stats(records: List[StepRecord], output_dir: Path, tau: float) -> None:
    stats_dir = output_dir / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(records)
    save_json(stats_dir / "summary_E_CI.json", summary)
    diffs = pairwise_diff(summary, tau)
    if diffs:
        with (stats_dir / "pairwise_differential_exposure.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(diffs[0].keys()))
            writer.writeheader()
            writer.writerows(diffs)
    if pd is not None:
        rows = records_to_rows(records)
        if rows:
            pd.DataFrame(rows).to_csv(stats_dir / "all_steps.csv", index=False, encoding="utf-8-sig")
    make_plots(records, stats_dir)
    print(f"统计结果已导出：{stats_dir}")


def make_plots(records: List[StepRecord], stats_dir: Path) -> None:
    if plt is None or not records:
        print("未检测到 matplotlib，跳过绘图。")
        return
    profiles = sorted(set(r.profile_id for r in records))
    # E 曲线：每个 profile 一张图，每条线一个维度。
    for pid in profiles:
        rs = [r for r in records if r.profile_id == pid]
        rs.sort(key=lambda x: x.timestamp)
        if not rs:
            continue
        plt.figure(figsize=(10, 6))
        xs = list(range(1, len(rs) + 1))
        for dim in DPS_DIMS:
            vals = []
            acc = 0.0
            for idx, r in enumerate(rs, start=1):
                acc += float(r.dps.get(dim, 0))
                vals.append(acc / idx)
            plt.plot(xs, vals, marker="o", label=dim)
        plt.title(f"Running E(D_i | A={pid})")
        plt.xlabel("Step")
        plt.ylabel("Running mean exposure")
        plt.legend()
        plt.tight_layout()
        plt.savefig(stats_dir / f"curve_E_{pid}.png", dpi=160)
        plt.close()
    # profile 均值柱状图。
    summary = summarize(records)
    for dim in DPS_DIMS:
        plt.figure(figsize=(10, 5))
        ps = list(summary.keys())
        ys = [summary[p]["E"][dim] for p in ps]
        plt.bar(ps, ys)
        plt.title(f"E({dim} | Profile)")
        plt.xlabel("Profile")
        plt.ylabel(f"Mean {dim}")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(stats_dir / f"bar_E_{dim}.png", dpi=160)
        plt.close()


def is_stable(records: List[StepRecord], window: int, eps: float) -> Tuple[bool, str]:
    if len(records) < 2 * window:
        return False, f"样本数 {len(records)} < {2*window}，暂不判断稳定。"
    prev = records[-2*window:-window]
    last = records[-window:]
    diffs = []
    for dim in DPS_DIMS:
        m1 = sum(r.dps.get(dim, 0.0) for r in prev) / window
        m2 = sum(r.dps.get(dim, 0.0) for r in last) / window
        diffs.append(abs(m2 - m1))
    max_diff = max(diffs)
    return max_diff < eps, f"最近两个窗口最大均值差={max_diff:.3f}；阈值={eps:.3f}。"


def select_profile(profiles: Dict[str, Any]) -> str:
    keys = list(profiles.keys())
    print("\n可选 Profile：")
    for idx, pid in enumerate(keys, start=1):
        print(f"  {idx}. {pid} - {profiles[pid].get('description', '')}")
    while True:
        raw = input("请选择 profile 编号或名称：").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(keys):
            return keys[int(raw) - 1]
        if raw in profiles:
            return raw
        print("输入无效。")


def run_experiment(config: Dict[str, Any], profiles_data: Dict[str, Any], rng: random.Random, output_dir: Path) -> None:
    profiles = profiles_data["profiles"]
    keywords = profiles_data.get("keywords", DEFAULT_KEYWORDS)
    app = input(f"App 名称 [{config['app']}]: ").strip() or config["app"]
    config["app"] = app
    profile_id = select_profile(profiles)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_app = app.replace("/", "_").replace("\\", "_")
    run_dir = output_dir / safe_app / profile_id / run_id
    screenshot_dir = run_dir / "screenshots"
    records: List[StepRecord] = []
    max_steps = int(config.get("max_steps", 30))
    delay = int(config.get("screenshot_delay_seconds", 3))
    print(f"\n开始实验：App={app}, Profile={profile_id}, max_steps={max_steps}")
    print("命令提示：每步标注前可按 Enter 继续；标注中输入 q 中断；主菜单可撤回最后一步。")

    while len(records) < max_steps:
        step = len(records) + 1
        action, category, keyword, instruction = generate_action(profiles[profile_id], keywords, rng, app, delay)
        print("\n" + "=" * 72)
        print(f"App: {app}")
        print(f"Profile: {profile_id}")
        print(f"Step: {step:03d}")
        print(f"Action: {action}")
        print(f"Category: {category}")
        print(f"Keyword: {keyword}")
        print(f"Instruction: {instruction}")
        print("=" * 72)
        cmd = input("执行完人工操作后按 Enter 截图并标注；输入 undo 撤回上一步；输入 stop 结束本 profile：").strip().lower()
        if cmd == "stop":
            break
        if cmd == "undo":
            if records:
                removed = records.pop()
                write_records(records, run_dir)
                print(f"已撤回 step {removed.step:03d}。")
            else:
                print("暂无可撤回步骤。")
            continue

        screenshot_path = None
        if bool(config.get("screenshot_after_each_action", True)):
            screenshot_path = take_screenshot(screenshot_dir / f"step_{step:03d}.png", delay)
        try:
            dps = prompt_dps()
        except KeyboardInterrupt:
            print("本步已取消，未写入记录。")
            continue
        note = input("可选备注，例如特殊页面/异常情况，直接 Enter 跳过：").strip()
        rec = StepRecord(
            app=app, profile_id=profile_id, step=step,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            action=action, category=category, keyword=keyword,
            instruction=instruction, screenshot_path=screenshot_path,
            dps=dps, note=note,
        )
        records.append(rec)
        write_records(records, run_dir)
        print("本步已保存。当前本 profile 估计：")
        print_summary(summarize(records))
        stable, msg = is_stable(records, int(config.get("stability_window", 5)), float(config.get("stability_epsilon", 0.10)))
        print(f"稳定性检查：{msg}")
        if stable:
            ans = input("已达到稳定阈值。是否结束本 profile？[y/N]: ").strip().lower()
            if ans == "y":
                break
    print(f"\n本次实验结束，数据目录：{run_dir}")


def edit_profile(profiles_data: Dict[str, Any], profiles_path: Path) -> None:
    profiles = profiles_data["profiles"]
    pid = input("输入新/已有 profile_id：").strip()
    if not pid:
        return
    desc = input("description：").strip()
    print("请输入 behavior 概率。留空则使用默认 0。")
    behavior = {}
    for a in ["visit_homepage", "search_keyword", "browse_or_scroll_page", "decision_interface_visit"]:
        raw = input(f"  {a}: ").strip()
        behavior[a] = float(raw or 0)
    print("请输入 keyword category 概率。留空则使用默认 0。")
    key_words = {}
    for c in profiles_data.get("keywords", DEFAULT_KEYWORDS).keys():
        raw = input(f"  {c}: ").strip()
        key_words[c] = float(raw or 0)
    normalize_dist(behavior)
    normalize_dist(key_words)
    profiles[pid] = {"description": desc, "behavior": behavior, "key_words": key_words}
    save_json(profiles_path, profiles_data)
    print("profile 已保存。")


def main() -> None:
    base = Path.cwd()
    config_path, profiles_path = ensure_default_files(base)
    config = load_json(config_path)
    profiles_data = load_json(profiles_path)
    output_dir = Path(config.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(config.get("random_seed", 20260603)))

    print("\nDark Pattern Differential Exposure Workbench")
    print(f"配置文件：{config_path}")
    print(f"Profile 文件：{profiles_path}")
    print(f"输出目录：{output_dir.resolve()}")
    if pyautogui is None:
        print("提示：未安装 pyautogui，自动截图不可用；可手动截图。安装：pip install pyautogui pillow")

    while True:
        print("\n主菜单")
        print("1. 开始/继续一次 profile 实验")
        print("2. 查看全部统计与 Differential Exposure")
        print("3. 导出统计、曲线和 CSV")
        print("4. 新增/修改 profile")
        print("5. 查看 DPS 标注指南")
        print("6. 退出")
        choice = input("请选择：").strip()
        if choice == "1":
            run_experiment(config, profiles_data, rng, output_dir)
        elif choice == "2":
            records = load_all_records(output_dir)
            summary = summarize(records)
            print_summary(summary)
            diffs = pairwise_diff(summary, float(config.get("differential_tau", 0.20)))
            if diffs:
                print("\n=== Pairwise Differential Exposure，超过 tau 的项 ===")
                any_hit = False
                for row in diffs:
                    if row["exceeds_tau"]:
                        any_hit = True
                        print(f"{row['profile_a']} vs {row['profile_b']} | {row['dimension']}: diff={row['diff_a_minus_b']:.3f}, abs={row['abs_diff']:.3f}")
                if not any_hit:
                    print("暂无维度超过 tau。")
        elif choice == "3":
            records = load_all_records(output_dir)
            export_stats(records, output_dir, float(config.get("differential_tau", 0.20)))
        elif choice == "4":
            edit_profile(profiles_data, profiles_path)
        elif choice == "5":
            for dim in DPS_DIMS:
                g = DPS_GUIDE[dim]
                print(f"\n{dim} - {g['name']}\n标尺：{g['scale']}\n判别：{g['rule']}\n例子：{g['examples']}")
        elif choice == "6":
            print("退出。")
            return
        else:
            print("输入无效。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n已中断，已保存的步骤不会丢失。")
        sys.exit(0)