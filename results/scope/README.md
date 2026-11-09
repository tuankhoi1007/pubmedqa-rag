# Ngưỡng out-of-scope

`RAG_MIN_SCORE` mặc định là **0.68**. `answer()` chỉ chặn khi top-1 score **nhỏ hơn** ngưỡng. Score bằng 0.68 vẫn gọi Gemini. Câu bị chặn trả `out_of_scope`, không gọi Gemini và không lấy slot rate limiter.

Ngưỡng được chọn trên 200 câu train đầu của PubMedQA `pqa_labeled` và 50 câu off-topic trong `tests/data/off_topic.txt`. Đo bằng `RAGEngine.retrieve()`, không gọi Gemini.

| Ngưỡng | In-domain bị chặn | Off-topic bị chặn |
|---:|---:|---:|
| 0.40–0.52 | 0% (0/200) | 0% (0/50) |
| 0.54 | 0.5% (1/200) | 4% (2/50) |
| 0.56 | 0.5% (1/200) | 18% (9/50) |
| 0.58 | 0.5% (1/200) | 30% (15/50) |
| 0.60 | 0.5% (1/200) | 48% (24/50) |
| 0.62 | 0.5% (1/200) | 66% (33/50) |
| 0.64 | 0.5% (1/200) | 80% (40/50) |
| 0.66 | 0.5% (1/200) | 88% (44/50) |
| **0.68** | **0.5% (1/200)** | **92% (46/50)** |
| 0.70 | 1.0% (2/200) | 98% (49/50) |

0.68 nằm dưới mức 1% in-domain bị chặn nhầm. Câu in-domain duy nhất dưới ngưỡng là index 106, score 0.5313.

Xác nhận trên tập chưa dùng để chọn ngưỡng: train index 200–399, 30 paraphrase (seed 0) của 200 câu đầu, và 30 câu off-topic mới.

| Nhóm | n | 0.66 | 0.68 | 0.70 |
|---|---:|---:|---:|---:|
| In-domain held-out | 200 | 0% (0) | 0% (0) | 0.5% (1) |
| Paraphrase | 30 | 0% (0) | 0% (0) | 0% (0) |
| Off-topic held-out | 30 | 100% (30) | 100% (30) | 100% (30) |

Số liệu đầy đủ: `results/scope/summary.json` và `results/scope/heldout.json`.
