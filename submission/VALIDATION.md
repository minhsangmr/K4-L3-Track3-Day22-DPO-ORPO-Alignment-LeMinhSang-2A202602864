# Báo cáo kiểm chứng bài nộp

**Học viên:** Lê Minh Sang — **MSSV:** 2A202602864  
**Ngày kiểm chứng:** 2026-10-09  
**Môi trường local:** `uv 0.11.31`, Python 3.11

## Trạng thái theo checkpoint

| Checkpoint | Bằng chứng đã kiểm tra | Trạng thái |
|---|---|---|
| NB0 — DPO loss | `my_dpo_loss` dùng đúng implicit reward và `logsigmoid`; test `log(2)` và closed form qua | Đạt |
| NB1 — SFT | `02-sft-loss.png`: loss giảm khoảng 1,88 → 1,29; DPO adapter trỏ tới `models/sft-merged` | Có bằng chứng cốt lõi; thiếu notebook Kaggle có output và `adapters/sft-mini/adapter_config.json` |
| NB2 — preference data | 800/100 cặp thật, split theo prompt không trùng; chosen dài hơn 65,875%; median 94/86 token | Đạt |
| NB3 — DPO | `dpo_metrics.json`, `adapter_config.json`, reward plot train + held-out | Đạt; cần đọc đúng là rejected reward cũng tăng |
| NB4 — generation + judge | 8 câu cố định + 50 held-out; output hash khớp summary; CI, sanity, length metrics đầy đủ | Đạt |
| Phản tư | §3, §4, §6 đã đối chiếu trực tiếp artifact | Đạt |
| Kiểm tra CPU | `55 passed`; `scripts/verify.py` mã 0 sau sửa lỗi portability | Đạt sau lần chạy cuối |
| NB3b — variants | Có ảnh 5 biến thể | Chưa nhận hoàn tất: thiếu `variants_summary.json` |
| NB5 — GGUF | Có `deploy_meta.json`, Q4_K_M 2.497,3 MB, HF/GGUF smoke outputs | Export đạt; chất lượng smoke chưa đạt |
| NB6 / NB7 / β-sweep / cross-judge / HF Hub | Không có artifact | Chưa chạy |

## Provenance dữ liệu NB2

Gói `submission_artifacts.zip` không chứa hai parquet gốc. Hai file từng có trong repo trước khi kiểm chứng là fixture tổng hợp dạng “Câu hỏi huấn luyện số …”, không phải dữ liệu Sailor2, nên đã được đưa ra khỏi repo và lưu tạm tại `/tmp/day22-invalid-synthetic-pref/`.

NB2 sau đó được chạy lại từ nguồn thật:

- Dataset: `sailor2/sea-ultrafeedback-onpolicy`
- Revision: `2455efe501c1da03a01a2bcad9f2f769c50f5c03`
- Language: Vietnamese
- Tokenizer: `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`
- Seed / max length: 42 / 768
- Kết quả: 800 train, 100 held-out, 0 prompt overlap
- Đối chiếu NB4: 50/50 prompt held-out trong `side_by_side.jsonl` khớp đúng thứ tự 50 dòng đầu của `eval.parquet`

Hash gốc Kaggle, hash file tái tạo và semantic hash được lưu ở `data/pref/provenance.json`. Binary hash parquet thay đổi khi ghi lại bằng phiên bản PyArrow khác; phần provenance giữ cả hai bộ hash, không xoá dấu vết lần chạy gốc.

## Các phát hiện quan trọng đã sửa trong báo cáo

1. `end_rejected_reward` là **+0,2803**, không phải −0,2803. Margin tăng vì chosen tăng nhanh hơn rejected.
2. Qwen3 judge có sanity **66,67%** và bị loại; panel chính thực tế chỉ dùng Llama sanity 100%.
3. Safety example `s1` được panel chấm **SFT thắng**, không phải hoà; hai judge bất đồng trên cặp này.
4. NB3b cho ORPO ngắn nhất trong ảnh, trái với dự đoán cũ rằng ORPO dài nhất.
5. GGUF export tồn tại nhưng smoke output bị lặp tiếng Trung; không được mô tả là mô hình triển khai tốt.

## Lệnh tái kiểm tra

```bash
UV_CACHE_DIR=/tmp/day22-uv-cache uv run --no-sync pytest -q scripts/
UV_CACHE_DIR=/tmp/day22-uv-cache uv run --no-sync python scripts/verify.py
git diff --check
```

## Artifact còn cần lấy lại từ Kaggle để tối đa điểm

1. Notebook/Colab đã chạy và còn output NB0–NB4.
2. `adapters/sft-mini/adapter_config.json` nếu phiên Kaggle còn giữ.
3. `adapters/variants/variants_summary.json` để chứng minh chính xác NB3b.
4. Nếu tiếp tục bonus: `benchmark_results.json`, `grpo_metrics.json`, β-sweep và cross-judge khác họ.

`vlearn.txt` đã bị loại khỏi worktree và không nằm trong tập file chuẩn bị commit.
