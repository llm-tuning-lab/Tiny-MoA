"""
Tiny MoA CLI 진입점
==================
python -m tiny_moa.main [--interactive]
"""

import argparse
import warnings

# Suppress ResourceWarning: unclosed file <_io.TextIOWrapper ...>
warnings.filterwarnings("ignore", category=ResourceWarning)
# Specific filter for the likely Windows 'nul' issue (RegEx escaped)
warnings.filterwarnings("ignore", message=r"unclosed file <_io.TextIOWrapper name='nul'", category=ResourceWarning)
# Catch-all for cp949 encoding issue often seen on Korean Windows
warnings.filterwarnings("ignore", message=r".*cp949.*", category=ResourceWarning)

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from tiny_moa.orchestrator import TinyMoA, interactive_mode

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Tiny MoA - GPU Poor를 위한 AI 군단",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  python -m tiny_moa.main                    # 기본 테스트 실행
  python -m tiny_moa.main --interactive      # 대화형 모드
  python -m tiny_moa.main --query "피보나치 함수 작성해줘"
        """,
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="대화형 모드 실행",
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="단일 쿼리 실행",
    )
    
    parser.add_argument(
        "--thinking",
        action="store_true",
        default=False,
        help="LFM Thinking 모델 사용 (실험 중)",
    )

    parser.add_argument(
        "--show-thinking",
        action="store_true",
        default=False,
        help="Thinking Process 출력 여부 (Thinking 모델 사용 시)",
    )

    parser.add_argument(
        "--tui",
        action="store_true",
        default=False,
        help="Tiny Cowork TUI 모드 실행",
    )

    parser.add_argument(
        "--n-ctx",
        type=int,
        default=4096,
        help="Context Window Size (default: 4096)",
    )
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    elif args.query:
        if not args.tui:
            print("🌐 Translation Pipeline 활성화")
            print("🤖 Tiny MoA 초기화 중...")
        moa = TinyMoA(
            use_thinking=args.thinking, 
            show_thinking=args.show_thinking,
            n_ctx=args.n_ctx
        )
        
        if args.tui:
            result = moa.run_cowork_flow(args.query)
            console.print("\n[bold green]✅ Cowork 작업 완료![/bold green]")
            console.print(Panel(Markdown(result), title="최종 결과 리포트", border_style="green"))
        else:
            moa.chat(args.query)
    else:
        console.print("[bold]🧪 Tiny MoA 기본 테스트[/bold]\n")
        
        moa = TinyMoA(
            use_thinking=args.thinking, 
            show_thinking=args.show_thinking,
            n_ctx=args.n_ctx
        )
        
        test_queries = [
            "안녕하세요! 반갑습니다.",
            "피보나치 수열의 10번째 항을 구하는 Python 함수를 작성해줘.",
            "1부터 100까지의 합은?",
        ]
        
        for query in test_queries:
            console.print(f"\n{'='*60}")
            moa.chat(query)


if __name__ == "__main__":
    main()
