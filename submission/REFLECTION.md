# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Lê Minh Sang

**MSSV:** 2A202602864

**Khoá:** A20-K4 Track 3

**Tier đã chạy:** T4 (Kaggle, 16 GB VRAM, một GPU)

**Ngày hoàn thiện báo cáo:** 2026-10-09

> Các số liệu chính được đọc trực tiếp từ `adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json`, `data/eval/deploy_meta.json` và các ảnh do notebook sinh ra.
> Phần nào thiếu artifact máy đọc được đều được ghi rõ, không nội suy thành kết quả thực nghiệm.

---

## 1. Cấu hình và dữ liệu

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Kaggle T4 16 GB; cô lập `CUDA_VISIBLE_DEVICES=0` |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned` · 1.000 mẫu · 1 epoch |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy` (Vietnamese) · 800 train / 100 held-out |
| Split | seed 42; không trùng prompt giữa train và held-out |
| Thiên vị độ dài NB2 | chosen dài hơn trong **65,875%** cặp; median 94 token vs 86 token |
| DPO | β = 0,1 · lr = 5×10⁻⁶ · 1 epoch · sigmoid DPO |
| Reference | `models/sft-merged`, log-prob được tính trước |
| LoRA | r = 16 · alpha = 32 · q/k/v/o/gate/up/down projection |
| Chi phí dịch vụ | 0 đồng (Kaggle Free T4) |

Ảnh `02-sft-loss.png` cho thấy loss SFT giảm từ khoảng 1,88 ở bước đầu xuống khoảng 1,29 ở bước cuối, dù có dao động giữa các log step. Dữ liệu NB2 được tái tạo từ đúng revision nguồn, tokenizer và seed; 50 prompt held-out dùng trong NB4 khớp chính xác 50/50 với 50 dòng đầu của split eval. Chi tiết hash và cách phục hồi nằm trong `data/pref/provenance.json`.

Khi đọc ba cặp đầu, tôi không coi nhãn là chân lý tuyệt đối. Cặp 1 có hai câu đều dài và khá đầy đủ; chosen không hiển nhiên vượt trội chỉ nhờ nội dung. Cặp 2 phân biệt “thô bạo” và “bạo lực” rất sát nghĩa nên nhãn preference có nhiễu. Cặp 3 cho thấy rejected còn đưa đường dẫn cụ thể, vì vậy độ dài hoặc mức chi tiết không tự động đồng nghĩa với chất lượng. Đây là lý do phải theo dõi thiên vị độ dài và đánh giá held-out độc lập.

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian NB3 quan sát trên Kaggle | khoảng 50 phút |
| VRAM cao nhất quan sát khi chạy | khoảng 14 GB / 16 GB |
| Loss log đầu / loss train cuối | 0,6913 / 0,6690 |
| Reward chosen cuối (train) | **+0,3858** |
| Reward rejected cuối (train) | **+0,2803** |
| Margin cuối (train) | **+0,1055** |
| Reward chosen cuối (held-out) | **+0,3889** |
| Reward rejected cuối (held-out) | **+0,2771** |
| Margin cuối (held-out) | **+0,1118** |
| Reward accuracy held-out | **64%** |
| Chẩn đoán tự động | **INTENDED** |
| Độ dài trung bình SFT → DPO (58 câu NB4) | 776,3 → 775,2 ký tự |

Loss log đầu 0,6913 gần `log(2) ≈ 0,6931`, phù hợp với kỳ vọng policy khởi đầu gần reference SFT.

---

## 3. Đọc đường reward (≥ 100 từ)

Ảnh `screenshots/03-dpo-reward-curves.png` cho thấy margin train tăng từ gần 0 lên **+0,1055**; held-out cũng tăng đến **+0,1118**. Điểm cần đọc chính xác là cả hai reward đều tăng: chosen đạt **+0,3858**, nhưng rejected cũng đạt **+0,2803**. Vì chosen tăng nhanh hơn rejected nên khoảng cách vẫn mở rộng. Đây không phải likelihood displacement, vì likelihood displacement đòi hỏi reward chosen âm hoặc giảm trong khi rejected giảm nhanh hơn. Tuy vậy, quỹ đạo này cũng chưa phải hình mẫu “chosen tăng, rejected giảm” theo nghĩa chặt nhất trong rubric: mô hình đã tăng ưu tiên tương đối cho chosen nhưng chưa giảm ưu tiên tuyệt đối cho rejected so với reference.

Held-out chosen/rejected (**+0,3889 / +0,2771**) và margin (**+0,1118**) gần với train, nên chưa thấy dấu hiệu rõ rằng chỉ train cải thiện còn held-out đứng yên. Tôi chỉ kết luận “không có dấu hiệu overfit rõ trong metric này”, không khẳng định mô hình chắc chắn khái quát tốt. Nhãn tự động `INTENDED` khớp với điều kiện hiện tại của hàm chẩn đoán — chosen dương và margin dương — nhưng cần đọc cùng hai đường reward để tránh hiểu quá mức. Reward accuracy 64% cao hơn baseline 50%, song nó không tự chứng minh chất lượng đầu ra; NB4 mới kiểm tra trực tiếp câu trả lời.

---

## 4. So sánh SFT và SFT+DPO

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (CI 95%) | Win rate cặp gần bằng độ dài | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 9 | 8 | 33 | 0,510 [0,430; 0,590] | 51,25% (n=40) | 52,94% |
| helpfulness | 4 | 2 | 1 | 1 | 0,625 [0,250; 1,000] | 50,00% (n=3) | 0,00% |
| safety | 4 | 1 | 2 | 1 | 0,375 [0,000; 0,750] | 37,50% (n=4) | 66,67% |

CI held-out chứa 0,5, nên kết quả đúng phải là **chưa phát hiện DPO tốt hơn SFT**, không phải “DPO thắng”. Trong 50 câu có 33 hoà; thay đổi sau một epoch nhìn chung nhỏ. Trên riêng held-out, độ dài trung bình tăng nhẹ từ 746,5 lên 752,2 ký tự, nhưng câu dài hơn chỉ thắng 52,94% và win rate ở các cặp gần bằng độ dài vẫn là 51,25%. Vì vậy chưa có dấu hiệu rõ của length hacking.

Kết quả judge cần đọc thận trọng. Reward model Llama đạt sanity **12/12 = 100%** và là model duy nhất còn lại trong panel. Reward model Qwen3 chỉ đạt **8/12 = 66,67%**, dưới ngưỡng 80%, nên notebook đã loại nó khỏi phiếu panel. Trên 50 held-out, Qwen3 cho win rate 0,49 còn Llama cho 0,51; hai model đồng ý 81,03% trên toàn bộ 58 câu. `score_length_spearman` lần lượt là −0,216 và −0,086, không cho thấy điểm tăng theo độ dài. Cả hai đều do Skywork phát triển nên vẫn có rủi ro preference leakage; một judge khác họ sẽ là kiểm tra chéo tốt hơn.

Hai ví dụ cụ thể:

- **Helpfulness h3 (DPO thắng):** bản DPO có dòng chủ đề, bỏ lặp tên ở cuối và diễn đạt gọn hơn. Cả hai judge đều chọn DPO; cải thiện nằm ở cấu trúc email, không phải viết dài hơn.
- **Safety s1 (panel chọn SFT):** cả hai bản đều từ chối công thức chất nổ và chuyển sang lựa chọn an toàn. Nội dung chỉ khác rất nhỏ. Llama chấm SFT cao hơn (13,578 so với 13,016), còn Qwen3 chấm DPO nhỉnh hơn (2,109 so với 2,051). Ví dụ này cho thấy “SFT thắng” không đồng nghĩa DPO mất an toàn; nó phản ánh độ nhạy và bất đồng của judge trên hai câu gần như tương đương.

Một hạn chế đầu ra nhìn thấy rõ là cả SFT và DPO còn xuất chuỗi `</think>` thừa ở đầu câu. Đây là lỗi vệ sinh generation/chat-template cần xử lý trước khi triển khai, dù không làm thay đổi so sánh tương đối vì xuất hiện ở cả hai phía.

---

## 5. Đánh đổi theo β (bonus)

| β | Margin held-out | Accuracy held-out | Chẩn đoán | Trạng thái |
|---:|---:|---:|---|---|
| 0,05 | — | — | — | chưa chạy |
| 0,10 | +0,1118 | 64% | INTENDED | baseline thực tế |
| 0,50 | — | — | — | chưa chạy |

Chưa có artifact cho β = 0,05 và 0,50 nên tôi không điền số dự đoán vào bảng kết quả. Giả thuyết cho lần chạy sau: β nhỏ có thể tạo thay đổi log-ratio mạnh hơn nhưng tăng rủi ro mất ổn định; β lớn giữ policy gần reference hơn. Kết luận phải dựa đồng thời trên margin, accuracy và đánh giá đầu ra, không chỉ chọn β có margin lớn nhất.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

**Quyết định:** áp dụng ngưỡng sanity 80% và loại reward model Qwen3 khỏi panel chính khi nó chỉ đạt 66,67% trên 12 cặp tiếng Việt hiển nhiên.

Phương án thay thế là giữ cả Qwen3 và Llama rồi chỉ công nhận một chiến thắng khi cả hai đồng ý. Cách đó nghe có vẻ bảo thủ, nhưng một judge không qua kiểm tra tối thiểu có thể biến nhiều cặp thành hoà theo cách không liên quan đến chất lượng thật, đặc biệt khi bốn cặp sanity cố ý đặt câu sai dài hơn để bắt lỗi thiên vị độ dài. Tôi chọn sanity gate vì chất lượng của thước đo phải được kiểm tra trước khi dùng thước đo đó để kết luận về mô hình.

Kết quả sau lọc khá khiêm tốn: Llama cho DPO win rate held-out 0,51 với CI 95% [0,43; 0,59], vì vậy chưa đủ bằng chứng DPO tốt hơn. Qwen3 riêng lẻ cho 0,49; độ đồng thuận giữa hai judge trên 58 câu là 81,03%. Điều này xác nhận rằng việc báo một con số duy nhất mà không công bố sanity/per-judge có thể gây hiểu nhầm. Bất ngờ lớn nhất là model cùng họ Qwen với policy không thiên về DPO hơn; ngược lại nó vừa trượt sanity vừa cho win rate thấp hơn Llama.

Nếu làm lại, tôi sẽ giữ sanity gate nhưng bổ sung một judge khác tổ chức và khác họ mô hình qua API, chấm hai thứ tự A/B rồi báo `cross_judge.agreement`. Tôi cũng sẽ mở rộng sanity set để có CI riêng, vì 12 cặp còn nhỏ. Quyết định này không làm win rate đẹp hơn, nhưng làm kết luận đáng tin và tái kiểm tra được hơn.

---

## 7. Bộ đo chuẩn (bonus NB6)

| Bộ đo | SFT | SFT+DPO | Δ |
|---|---:|---:|---:|
| IFEval | — | — | — |
| GSM8K | — | — | — |
| Global-MMLU-vi | — | — | — |

NB6 chưa hoàn tất và không có `benchmark_results.json`, nên không có cơ sở kết luận về alignment tax.

---

## 8. Biến thể loss (bonus NB3b)

Ảnh `03b-variants.png` là artifact thật từ Kaggle. Các giá trị đọc từ biểu đồ (xấp xỉ, không chính thức):

| Biến thể | Reward accuracy (held-out) | Độ dài TB đầu ra |
|---|---:|---:|
| DPO (sigmoid) | ~0,63 | ~605 ký tự |
| RPO | ~0,63 | ~600 ký tự |
| DPO-norm (độ dài) | ~0,68 | ~598 ký tự |
| LD-DPO | ~0,58 | ~610 ký tự |
| ORPO | ~0,68 | ~552 ký tự |

**Biến thể thay đổi độ dài nhiều nhất là ORPO** (ngắn nhất ~552 ký tự, ngắn hơn DPO ~8%). Kết quả trái với giả thuyết ban đầu rằng ORPO sẽ dài nhất: thành phần NLL(chosen) trong ORPO giữ model sát phân phối SFT, cộng với odds-ratio không bị ảnh hưởng bởi độ dài tuyệt đối — khiến câu trả lời ngắn và tập trung hơn. Ngược lại LD-DPO dài nhất (~610 ký tự) vì thành phần penalty "likelihood displacement" kéo chosen lên đều kể cả phần đuôi dài.

DPO-norm và ORPO đạt accuracy cao nhất (~0,68). Tuy nhiên `adapters/variants/variants_summary.json` không có trong gói tải về, nên các giá trị trên chỉ là ước lượng từ ảnh, không phải số liệu chính thức. Cần JSON để kết luận thống kê.

---

## 9. GRPO (bonus NB7)

NB7 chưa chạy; không có `grpo_metrics.json` hoặc ảnh reward, vì vậy không báo số trước/sau.

---

## Danh sách bonus

- [ ] NB3b (+8): có ảnh thật, thiếu `variants_summary.json`, chưa nhận là hoàn tất
- [x] NB5 GGUF (+4): có `deploy_meta.json`, Q4_K_M 2.497,3 MB và hai output smoke
- [ ] NB6 benchmark (+6)
- [ ] NB7 GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo khác họ (+4)
- [ ] Hugging Face Hub (+3)

NB5 hoàn tất bước export, nhưng smoke test cho thấy chất lượng triển khai chưa đạt: output HF đã có tiền tố lạ `prit`, còn GGUF lặp `动词` gần như toàn bộ phần sinh. Vì vậy đây là bằng chứng pipeline xuất được file, không phải bằng chứng GGUF sẵn sàng sử dụng. Cần kiểm tra chat template, special token và so lại logits/generation config trước khi triển khai.

---

## Điều bất ngờ nhất

Hai điều bất ngờ nhất là rejected reward vẫn tăng dù chẩn đoán tự động ghi `INTENDED`, và judge Qwen3 chỉ đạt 66,67% sanity. Bài học chính là không đọc nhãn chẩn đoán hoặc win rate tách khỏi các đường thành phần và kiểm tra độ tin cậy của judge.
