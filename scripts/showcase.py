#!/usr/bin/env python

import json
import argparse
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.tree import Tree
from rich.table import Table
from rich import box
from rich.text import Text
from rich.style import Style

# Gruvbox Dark 调色板
GRUVBOX = {
    "bg0": "#282828",  # 背景色
    "bg1": "#3c3836",  # 深一级背景
    "bg2": "#504945",  # 更深背景
    "bg3": "#665c54",  # 最深背景
    "fg0": "#fbf1c7",  # 前景色
    "fg1": "#ebdbb2",  # 主要文字
    "fg2": "#d5c4a1",  # 次要文字
    "red": "#fb4934",  # 红色
    "green": "#b8bb26",  # 绿色
    "yellow": "#fabd2f",  # 黄色
    "blue": "#83a598",  # 蓝色
    "purple": "#d3869b",  # 紫色
    "aqua": "#8ec07c",  # 青色
    "orange": "#fe8019",  # 橙色
    "gray": "#928374",  # 灰色
}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="可视化查看轨迹数据的工具 (Gruvbox Dark 主题)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s -f data.json -i 0
  %(prog)s --file_path checkpoints/SearchRL/gem-search-grpo-qwen3-4b/chat_completions/global_steps_16.json --case_index 54
  %(prog)s -f data.json -i 10 --no-metadata
        """,
    )

    parser.add_argument("-f", "--file_path", type=str, default="checkpoints/SearchRL/gem-search-grpo-qwen3-4b/chat_completions/global_steps_16.json", help="JSON 数据文件路径 (默认: checkpoints/SearchRL/gem-search-grpo-qwen3-4b/chat_completions/global_steps_16.json)")

    parser.add_argument("-i", "--case_index", type=int, default=54, help="要查看的样本索引 (默认: 54)")

    parser.add_argument("--no_metadata", action="store_true", help="不显示元数据部分")

    parser.add_argument("--no_trajectory", action="store_true", help="不显示轨迹部分")

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 初始化控制台，设置宽度为 None 以避免截断
    console = Console(width=None)

    # 读取数据
    try:
        with open(args.file_path, "r") as f:
            raw_data = json.load(f)
        
        # Handle both old list format and new dictionary format
        if isinstance(raw_data, dict) and "chat_completions" in raw_data:
            data_list = raw_data["chat_completions"]
            stats = {k: v for k, v in raw_data.items() if k != "chat_completions"}
        else:
            data_list = raw_data
            stats = None

        if args.case_index >= len(data_list) or args.case_index < 0:
            console.print(f"[bold {GRUVBOX['red']}]错误: 索引 {args.case_index} 超出范围 (0-{len(data_list)-1})[/bold {GRUVBOX['red']}]")
            return

        data = data_list[args.case_index]

    except FileNotFoundError:
        console.print(f"[bold {GRUVBOX['red']}]错误: 文件未找到 '{args.file_path}'[/bold {GRUVBOX['red']}]")
        return
    except json.JSONDecodeError:
        console.print(f"[bold {GRUVBOX['red']}]错误: JSON 格式错误[/bold {GRUVBOX['red']}]")
        return
    except Exception as e:
        import traceback
        traceback.print_exc()
        console.print(f"[bold {GRUVBOX['red']}]错误: {str(e)}[/bold {GRUVBOX['red']}]")
        return

    # 显示文件信息
    console.rule(f"[{GRUVBOX['orange']}]📁 File Info[/{GRUVBOX['orange']}]", style=GRUVBOX["orange"])
    console.print()
    console.print(f"[{GRUVBOX['aqua']}]File Path:[/{GRUVBOX['aqua']}] {args.file_path}")
    console.print(f"[{GRUVBOX['aqua']}]Case Index:[/{GRUVBOX['aqua']}] {args.case_index}")
    console.print(f"[{GRUVBOX['aqua']}]Total Cases:[/{GRUVBOX['aqua']}] {len(data_list)}")
    
    if stats:
        console.print()
        console.print(f"[{GRUVBOX['purple']}]Stats summary:[/{GRUVBOX['purple']}]")
        for k, v in stats.items():
            if isinstance(v, dict):
                # Count non-zero stats
                non_zero = {rk: rv for rk, rv in v.items() if rv > 0}
                console.print(f"  [{GRUVBOX['aqua']}]{k}:[/{GRUVBOX['aqua']}] {non_zero}")
            else:
                console.print(f"  [{GRUVBOX['aqua']}]{k}:[/{GRUVBOX['aqua']}] {v}")
    console.print()

    # 分离轨迹和元数据
    trajectory = data[:-1]
    metadata = data[-1]

    # ============ 显示元数据 ============
    if not args.no_metadata:
        console.rule(f"[{GRUVBOX['purple']}]📊 Metadata[/{GRUVBOX['purple']}]", style=GRUVBOX["purple"])
        console.print()

        # 使用 JSON 语法高亮显示元数据 - 设置背景色为 bg1
        metadata_json = json.dumps(metadata, indent=4, ensure_ascii=False)
        metadata_syntax = Syntax(
            metadata_json,
            "json",
            theme="gruvbox-dark",
            line_numbers=True,
            word_wrap=False,  # 禁用自动换行，完整显示
            background_color=GRUVBOX["bg1"],  # 设置为 Gruvbox 背景色
        )

        metadata_panel = Panel(
            metadata_syntax,
            title=f"[bold {GRUVBOX['aqua']}]Metadata Details[/bold {GRUVBOX['aqua']}]",
            border_style=GRUVBOX["aqua"],
            box=box.ROUNDED,
            style=Style(bgcolor=GRUVBOX["bg1"]),
            expand=False,  # 不限制宽度
        )
        console.print(metadata_panel)
        console.print()

    # ============ 显示轨迹 ============
    if not args.no_trajectory:
        console.rule(f"[{GRUVBOX['green']}]🔄 Trajectory[/{GRUVBOX['green']}]", style=GRUVBOX["green"])
        console.print()

        # 为每个轨迹步骤创建面板
        for idx, step in enumerate(trajectory, 1):
            # 创建树状结构展示字段
            tree = Tree(f"[bold {GRUVBOX['yellow']}]Step {idx}[/bold {GRUVBOX['yellow']}]", guide_style=Style(color=GRUVBOX["gray"], dim=True))

            for key, value in step.items():
                # 特殊字段高亮
                if key == "role":
                    key_style = f"bold {GRUVBOX['red']}"
                elif key == "action":
                    key_style = f"bold {GRUVBOX['green']}"
                elif key in ["content", "state"]:
                    key_style = f"bold {GRUVBOX['blue']}"
                else:
                    key_style = GRUVBOX["aqua"]

                branch = tree.add(f"[{key_style}]{key}[/{key_style}]")

                # 格式化值的显示
                if isinstance(value, dict):
                    value_json = json.dumps(value, indent=4, ensure_ascii=False)
                    value_syntax = Syntax(
                        value_json,
                        "json",
                        theme="gruvbox-dark",
                        background_color=GRUVBOX["bg1"],  # 设置为 Gruvbox 背景色
                    )
                    branch.add(value_syntax)
                elif isinstance(value, list):
                    value_json = json.dumps(value, indent=4, ensure_ascii=False)
                    value_syntax = Syntax(
                        value_json,
                        "json",
                        theme="gruvbox-dark",
                        background_color=GRUVBOX["bg1"],  # 设置为 Gruvbox 背景色
                    )
                    branch.add(value_syntax)
                elif isinstance(value, str):
                    # 完整显示所有文本，不省略
                    text = Text(value, style=GRUVBOX["fg1"])
                    branch.add(text)
                else:
                    text = Text(str(value), style=GRUVBOX["fg1"])
                    branch.add(text)

            # 将树放入面板 - 使用 Gruvbox 背景色
            step_panel = Panel(
                tree,
                title=f"[bold {GRUVBOX['yellow']}]Trajectory Step {idx}[/bold {GRUVBOX['yellow']}]",
                border_style=GRUVBOX["yellow"],
                box=box.DOUBLE,
                style=Style(bgcolor=GRUVBOX["bg1"]),  # 使用 Gruvbox 背景色
                expand=False,  # 不限制宽度
            )
            console.print(step_panel)
            console.print()

    console.rule(style=GRUVBOX["purple"])


if __name__ == "__main__":
    main()
