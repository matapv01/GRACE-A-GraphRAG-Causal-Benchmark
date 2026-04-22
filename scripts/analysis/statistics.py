import json
import argparse
from pathlib import Path
from rich.console import Console
from graphrag_benchmark.infrastructure.dataset_loader import DatasetLoader


def main():
    parser = argparse.ArgumentParser(
        description="Generate statistics for extracted and perturbed datasets."
    )
    args = parser.parse_args()

    console = Console()

    import glob
    import re

    dataset_file = "data/lcquad_test.json"
    clean_dir = Path("data/test_clean_subgraphs")
    perturb_dir = Path("data/test_perturbed_subgraphs")
    log_pattern = "logs_extract_test*.txt"

    console.print(f"Loading dataset for baseline total (MODE: TEST)...")
    try:
        loader = DatasetLoader(dataset_file)
        questions = loader.load_dataset()
        total_questions = len(questions)
    except Exception as e:
        console.print(f"[red]Could not load dataset {dataset_file}: {e}[/red]")
        total_questions = 5000  # Fallback

    clean_count = 0
    if clean_dir.exists():
        clean_count = len(list(clean_dir.glob("*.json")))

    processed_count = clean_count  # Fallback
    extract_errors = {}
    extract_error_samples = {}
    logged_error_count = 0
    logged_error_qids = set()

    log_files = glob.glob(log_pattern)
    existing_clean_files = (
        {f.stem for f in clean_dir.glob("*.json")} if clean_dir.exists() else set()
    )

    # Dùng dictionary lưu lỗi cuối cùng của mỗi câu hỏi
    actual_errors_by_qid = {}

    for lf in sorted(log_files):
        try:
            # Đọc ngược từ cuối file để tìm dòng tiến độ tqdm mới nhất
            with open(lf, "rb") as f:
                f.seek(0, 2)
                file_size = f.tell()
                read_size = min(5000, file_size)
                f.seek(file_size - read_size)
                tail_content = f.read().decode("utf-8", errors="ignore")
                matches = re.findall(rf"(\d+)/{total_questions}\s*\[", tail_content)
                if matches:
                    processed_count = max(
                        processed_count, int(matches[-1]), clean_count
                    )

            # Đọc toàn bộ file để gom nhóm các lỗi Exception
            with open(lf, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                error_matches = re.findall(
                    r"Error processing Question (\d+):\s*(.*)", content
                )
                for q_id, err_msg in error_matches:
                    # Nếu file này đã extract THÀNH CÔNG trong thực tế -> bỏ qua lỗi "ảo"
                    if q_id in existing_clean_files:
                        if q_id in actual_errors_by_qid:
                            del actual_errors_by_qid[q_id]
                        continue

                    err_msg = re.sub(r"\x1b\[.*?m", "", err_msg).strip()
                    err_msg = re.sub(r"\[/?red\]", "", err_msg)
                    short_err = (
                        err_msg.split(":")[0] if ":" in err_msg else err_msg[:50]
                    )
                    actual_errors_by_qid[q_id] = short_err

        except Exception:
            pass

    for q_id, short_err in actual_errors_by_qid.items():
        extract_errors[short_err] = extract_errors.get(short_err, 0) + 1
        extract_error_samples[short_err] = q_id
        logged_error_count += 1
        logged_error_qids.add(q_id)

    pending_count = total_questions - processed_count
    error_count = processed_count - clean_count

    perturb_total = 0
    perturb_success = 0
    perturb_error = 0
    error_types = {}
    perturb_error_samples = {}

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
                        short_err = (
                            err_msg.split(":")[0] if ":" in err_msg else err_msg[:50]
                        )
                        if "list index out of range" in short_err:
                            short_err = (
                                "Index out of range (Thường do chuỗi logic trống)"
                            )
                        elif "Could not find any edge leading to answer" in short_err:
                            short_err = (
                                "Không tìm thấy cạnh Answer (Failed Broken Graph)"
                            )

                        error_types[short_err] = error_types.get(short_err, 0) + 1
                        if short_err not in perturb_error_samples:
                            perturb_error_samples[short_err] = p_file.stem
                    else:
                        perturb_success += 1
            except Exception:
                perturb_error += 1
                error_types["Unreadable JSON"] = (
                    error_types.get("Unreadable JSON", 0) + 1
                )
                if "Unreadable JSON" not in perturb_error_samples:
                    perturb_error_samples["Unreadable JSON"] = p_file.stem

    # In kết quả
    console.print("\n[bold cyan]--- THỐNG KÊ TRẠNG THÁI DỮ LIỆU ĐÃ CÀO ---[/bold cyan]")
    console.print(
        f"Tổng số câu hỏi gốc cài đặt ban đầu: [bold]{total_questions}[/bold]"
    )
    console.print(
        "\n[bold yellow]Giai đoạn 1: Trích xuất (Extract Clean Subgraphs)[/bold yellow]"
    )
    console.print(
        f"- Thành công (Đã có file Clean): [bold green]{clean_count}[/bold green] ({((clean_count / total_questions) * 100) if total_questions else 0:.2f}%)"
    )
    console.print(
        f"- Lỗi / Dữ liệu rác (Đã chạy qua):[bold red]{error_count}[/bold red] ({((error_count / total_questions) * 100) if total_questions else 0:.2f}%)"
    )

    if error_count > 0:
        console.print("\n[dim]  Chi tiết nguyên nhân loại bỏ:[/dim]")
        silent_skips = error_count - logged_error_count
        if silent_skips > 0:
            empty_gold_path_sample = None
            if clean_dir.exists():
                existing_clean = {f.stem for f in clean_dir.glob("*.json")}
                for q in questions[:processed_count]:
                    if (
                        str(q.id) not in existing_clean
                        and str(q.id) not in logged_error_qids
                    ):
                        empty_gold_path_sample = str(q.id)
                        break

            sample_text = (
                f" [dim](Question ID: {empty_gold_path_sample})[/dim]"
                if empty_gold_path_sample
                else ""
            )
            console.print(
                f"    - Câu hỏi không thể tìm ra đường dẫn (Empty Gold Path): [red]{silent_skips} samples[/red]{sample_text}"
            )

        for k, v in sorted(
            extract_errors.items(), key=lambda item: item[1], reverse=True
        )[:5]:
            sample_id = extract_error_samples.get(k)
            sample_text = f" [dim](Question ID: {sample_id})[/dim]" if sample_id else ""
            console.print(f"    - {k}: [red]{v} samples[/red]{sample_text}")

    console.print(
        f"\n- Đang chờ (Chưa cào tới):       [bold magenta]{pending_count}[/bold magenta] ({((pending_count / total_questions) * 100) if total_questions else 0:.2f}%)"
    )

    console.print(
        "\n[bold yellow]Giai đoạn 2: Nhiễu loạn (Generate Perturbations)[/bold yellow]"
    )
    if perturb_total > 0:
        console.print(
            f"- Tổng số subgraphs đã duyệt qua: [bold]{perturb_total}[/bold] (từ {clean_count} file Clean)"
        )
        console.print(
            f"- Thành công (Sinh đủ 5 biểu đồ): [bold green]{perturb_success}[/bold green] ({(perturb_success / perturb_total) * 100:.2f}%)"
        )
        console.print(
            f"- Lỗi (Văng Log hoặc Null Answer):[bold red]{perturb_error}[/bold red] ({(perturb_error / perturb_total) * 100:.2f}%)"
        )

        if error_types:
            console.print(
                "\n[dim]  Chi tiết các lỗi thường gặp trong lúc Perturb:[/dim]"
            )
            # Sắp xếp lỗi phổ biến nhất lên đầu
            for k, v in sorted(
                error_types.items(), key=lambda item: item[1], reverse=True
            )[:5]:
                sample_text = (
                    f" [dim](VD: {perturb_error_samples.get(k, 'N/A')}.json)[/dim]"
                )
                console.print(f"    - {k}: [red]{v} samples[/red]{sample_text}")
    else:
        console.print("- [dim]Chưa thu thập được dữ liệu Perturb nào.[/dim]")


if __name__ == "__main__":
    main()
