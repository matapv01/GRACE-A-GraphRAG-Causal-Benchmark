import os
import json
from pathlib import Path
from rich.console import Console
from graphrag_benchmark.infrastructure.dataset_loader import DatasetLoader

def main():
    console = Console()
    
    # Load dataset for total count
    console.print("Loading dataset for baseline total...")
    try:
        loader = DatasetLoader("data/lcquad_train.json")
        questions = loader.load_dataset()
        total_questions = len(questions)
    except Exception as e:
        console.print(f"[red]Could not load dataset: {e}[/red]")
        total_questions = 24180 # Fallback for LC-QuAD 2.0 Train
        
    clean_dir = Path("data/clean_subgraphs")
    perturb_dir = Path("data/perturbed_subgraphs")
    
    clean_count = 0
    if clean_dir.exists():
        clean_count = len(list(clean_dir.glob("*.json")))
        
    extraction_failed = total_questions - clean_count
    
    perturb_total = 0
    perturb_success = 0
    perturb_error = 0
    error_types = {}
    
    if perturb_dir.exists():
        for p_file in perturb_dir.glob("*.json"):
            perturb_total += 1
            try:
                with open(p_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    # Perturb script writes {"error": "..."} when it fails
                    if "error" in data:
                        perturb_error += 1
                        err_msg = str(data["error"])
                        
                        # Rút gọn message lỗi để gom nhóm (grouping) vì ID có thể khác nhau
                        short_err = err_msg.split(":")[0] if ":" in err_msg else err_msg[:50]
                        if "list index out of range" in short_err:
                            short_err = "Index out of range (Thường do chuỗi logic trống)"
                        elif "Could not find any edge leading to answer" in short_err:
                            short_err = "Không tìm thấy cạnh Answer (Failed Broken Graph)"
                            
                        error_types[short_err] = error_types.get(short_err, 0) + 1
                    else:
                        perturb_success += 1
            except Exception:
                perturb_error += 1
                error_types["Unreadable JSON"] = error_types.get("Unreadable JSON", 0) + 1
                
    # In kết quả
    console.print("\n[bold cyan]--- THỐNG KÊ TRẠNG THÁI DỮ LIỆU ĐÃ CÀO ---[/bold cyan]")
    console.print(f"Tổng số câu hỏi gốc cài đặt ban đầu: [bold]{total_questions}[/bold]")
    console.print(f"\n[bold yellow]Giai đoạn 1: Trích xuất (Extract Clean Subgraphs)[/bold yellow]")
    console.print(f"- Thành công (Đã có file Clean): [bold green]{clean_count}[/bold green] ({((clean_count/total_questions)*100) if total_questions else 0:.2f}%)")
    console.print(f"- Đang chờ / Lỗi / Dữ liệu rác:   [bold red]{extraction_failed}[/bold red] ({((extraction_failed/total_questions)*100) if total_questions else 0:.2f}%)")
    console.print("[dim]  (Chú ý: Extract fail không tạo file nên con số này bao gồm cả những câu chưa chạy tới)[/dim]")
    
    console.print(f"\n[bold yellow]Giai đoạn 2: Nhiễu loạn (Generate Perturbations)[/bold yellow]")
    if perturb_total > 0:
        console.print(f"- Tổng số subgraphs đã duyệt qua: [bold]{perturb_total}[/bold] (từ {clean_count} file Clean)")
        console.print(f"- Thành công (Sinh đủ 5 biểu đồ): [bold green]{perturb_success}[/bold green] ({(perturb_success/perturb_total)*100:.2f}%)")
        console.print(f"- Lỗi (Văng Log hoặc Null Answer):[bold red]{perturb_error}[/bold red] ({(perturb_error/perturb_total)*100:.2f}%)")
        
        if error_types:
            console.print("\n[dim]  Chi tiết các lỗi thường gặp trong lúc Perturb:[/dim]")
            # Sắp xếp lỗi phổ biến nhất lên đầu
            for k, v in sorted(error_types.items(), key=lambda item: item[1], reverse=True)[:5]:
                console.print(f"    - {k}: [red]{v} samples[/red]")
    else:
        console.print("- [dim]Chưa thu thập được dữ liệu Perturb nào.[/dim]")

if __name__ == "__main__":
    main()
