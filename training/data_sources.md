# Nguồn dữ liệu fine-tune NLLB (en→vi, ja→vi)

Khoá cấu hình là tên trong `DATASETS` của `finetune_nllb_colab.ipynb`. URL đã kiểm tra truy cập được ngày 2026-10-28.

| Bộ dữ liệu | Khoá cấu hình | Cặp | URL tải | License / điều khoản | Mặc định |
|---|---|---|---|---|---|
| IWSLT'15 en-vi (Stanford NMT, TED talks) | `iwslt15_en_vi` | en-vi | https://paddlenlp.bj.bcebos.com/datasets/iwslt15.en-vi.tar.gz (MD5 `aca22dc3f90962e42916dbb36d8f3e8e`); dự phòng https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/ | TED Talks Usage Policy (CC BY-NC-ND 4.0); trang Stanford không ghi license riêng | bật |
| OpenSubtitles v2018 en-vi (OPUS) | `opensubtitles_en_vi` | en-vi | https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-vi.txt.zip | Không ghi license; OPUS yêu cầu dẫn nguồn http://www.opensubtitles.org/ và trích dẫn Lison & Tiedemann (2016) | bật |
| TED2020 en-vi (OPUS) | `ted2020_en_vi` | en-vi | https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/en-vi.txt.zip | TED Talks Usage Policy (CC BY-NC-ND 4.0); trích dẫn Reimers & Gurevych (2020) | bật |
| PhoMT (VinAI) | `phomt_en_vi` | en-vi | https://huggingface.co/datasets/vinai/PhoMT (gated, file `PhoMT.zip`; cần token HF và chấp nhận điều khoản) | **chỉ nghiên cứu**: chỉ dùng cho nghiên cứu/giáo dục, không phân phối lại | **tắt** |
| OpenSubtitles v2018 ja-vi (OPUS) | `opensubtitles_ja_vi` | ja-vi | https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/ja-vi.txt.zip | Như OpenSubtitles en-vi | bật |
| TED2020 ja-vi (OPUS) | `ted2020_ja_vi` | ja-vi | https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/ja-vi.txt.zip | TED Talks Usage Policy (CC BY-NC-ND 4.0) | bật |
| ALT – Asian Language Treebank (NICT) | `alt_ja_vi` | ja-vi | https://huggingface.co/datasets/mutiyama/alt (cấu hình `alt-parallel`, nhánh `refs/convert/parquet`, revision `78f393e562d2940de035355229a7438192a06231`) | CC BY 4.0 | bật |
| Pivot ja-en→vi (OpenSubtitles v2018 en-ja + bản dịch en→vi của NLLB gốc) | `USE_PIVOT_JA_EN` | ja-vi (tổng hợp) | https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/moses/en-ja.txt.zip | Như OpenSubtitles; phần tiếng Việt do NLLB (CC-BY-NC-4.0) sinh ra | tắt |

## Model dùng trong notebook

| Model | Vai trò | URL | License |
|---|---|---|---|
| NLLB-200-distilled-600M | Model gốc để fine-tune, sinh dữ liệu pivot | https://huggingface.co/facebook/nllb-200-distilled-600M | CC-BY-NC-4.0 (phi thương mại) |
| LaBSE | Lọc cặp câu theo cosine (≥ 0.7) | https://huggingface.co/sentence-transformers/LaBSE | Apache-2.0 |

## Ghi chú

- Tập test: IWSLT'15 dùng `tst2013`, ALT dùng split `test` chính thức, PhoMT dùng split `test`; OpenSubtitles giữ riêng
  các phim không dùng để train (phim = thư mục `lang/NĂM/ID` ở cột 1 của file `.ids`), tối đa 60 dòng mỗi phim nên 300 dòng
  test trải trên ≥ 5 phim; TED2020 giữ riêng phần cuối luồng đọc. Tập test được khử trùng lặp theo câu nguồn đã chuẩn hoá.
  Câu train trùng câu test (so sánh sau khi bỏ hoa/thường, dấu câu, khoảng trắng; cả phía nguồn lẫn đích) bị loại, kể cả khi
  chỉ xuất hiện làm câu ngữ cảnh.
- OpenSubtitles: chọn ngẫu nhiên nguyên phim trên toàn bộ file (lượt 1 chỉ đọc `.ids` để đếm số dòng mỗi phim), không chỉ
  lấy các dòng đầu (file sắp theo năm).
- Ranh giới tài liệu (để ghép câu trước làm ngữ cảnh): OpenSubtitles có file `.ids` (mỗi cặp file phụ đề là một tài liệu),
  ALT dùng `SNT.URLID`; TED2020, IWSLT'15 và PhoMT không có id tài liệu nên được coi là một luồng liên tục.
- IWSLT'15 bản Stanford đã tách token kiểu Moses (`It &apos;s`), notebook detokenize bằng `sacremoses`.
- Các bản IWSLT'15 trên Hugging Face (`IWSLT/mt_eng_vietnamese` dạng script, các bản parquet `thainq107/iwslt2015-en-vi`,
  `nguyenvuhuy/iwslt2015-en-vi`, `Muscaria/iwslt2015-en-vi`, `Angelectronic/IWSLT15_English_Vietnamese`) bị lỗi mã hoá ký tự
  tiếng Việt (ví dụ `học` thành `h�\udc8dc`) nên không dùng.
- Kích thước file zip OPUS: OpenSubtitles en-vi ~107 MB, ja-vi ~23 MB, en-ja ~66 MB; TED2020 en-vi ~24 MB, ja-vi ~25 MB.

# Nguồn dữ liệu fine-tune Whisper tiếng Nhật (ASR ja)

Khoá cấu hình là tên trong `DATASETS` của `finetune_whisper_ja_colab.ipynb`. Truy cập và điều khoản đã kiểm tra ngày 2026-10-28.

| Bộ dữ liệu | Khoá cấu hình | URL đọc dữ liệu (revision) | License / điều khoản | Cần đăng nhập HF | Mặc định |
|---|---|---|---|---|---|
| Common Voice 17.0, tiếng Nhật (Mozilla) | `common_voice_ja` | https://huggingface.co/datasets/fixie-ai/common_voice_17_0 (cấu hình `ja`, revision `34f78a43893414e7b6e271ba94c1d5e05f18b239`, parquet: train 10.039 / validation 6.261 / test 6.261 câu) | CC0-1.0; điều khoản Common Voice: không tìm cách xác định danh tính người nói. Từ 10/2025 Mozilla chỉ phát hành Common Voice qua Mozilla Data Collective (https://datacollective.mozillafoundation.org, cần tài khoản và chấp nhận điều khoản); repo `mozilla-foundation/common_voice_*` trên Hugging Face không còn file dữ liệu | không | bật |
| ReazonSpeech v2 (Reazon Research), tập `small` | `reazonspeech` | Trang chính thức (gated): https://huggingface.co/datasets/reazon-research/reazonspeech. Dữ liệu đọc từ bản parquet https://huggingface.co/datasets/japanese-asr/whisper_transcriptions.reazonspeech.small (revision `c74b52fc164cf7b64936ca62aee4336eac626739`, 15 file, 62.047 câu, 16 kHz) | CDLA-Sharing-1.0; **chỉ được dùng cho mục đích theo Điều 30-4 Luật Bản quyền Nhật Bản**. Bản parquet không khai báo license riêng; điều khoản của bộ dữ liệu chính thức được áp dụng. Notebook chỉ đọc dữ liệu khi tài khoản đã được cấp quyền ở trang chính thức (đã chấp nhận điều khoản) | **có** (token Read + chấp nhận điều khoản) | bật |

## Model dùng trong notebook

| Model | Vai trò | URL | License |
|---|---|---|---|
| Whisper small | Model gốc để fine-tune, mốc CER | https://huggingface.co/openai/whisper-small | Apache-2.0 |
| faster-whisper small (CTranslate2) | Mốc CER int8 giống app (`asr.model = "small"`) | https://huggingface.co/Systran/faster-whisper-small | MIT |

## Ghi chú

- Tập test Common Voice: split `test` chính thức (các split của Common Voice không trùng người nói), lấy đều từ 8 file test.
  Câu train của người nói (`client_id`) có mặt trong split test bị loại; câu train trùng nội dung câu test (so sánh sau
  chuẩn hoá NFKC, bỏ dấu câu/ký hiệu/khoảng trắng) cũng bị loại. Train dùng split `train` + `validation`.
- Tập test ReazonSpeech: các câu đầu của file parquet cuối (`train-00014-of-00015`), file này không dùng để train. ReazonSpeech
  không có id người nói hay chương trình (tên file là mã băm), nên chỉ tách được theo file; câu train trùng nội dung câu test bị loại.
- Trang chính thức ReazonSpeech chỉ có script tải (`reazonspeech.py`), không chạy được với thư viện `datasets` ≥ 4.0, nên
  notebook đọc bản parquet có cùng âm thanh và transcript. Câu chỉ là chữ trên màn hình (trong ngoặc `［…］`) bị loại.
- Mọi âm thanh được resample về 16 kHz mono; lọc độ dài 0,5–30 giây và nhãn ≤ 440 token.
- Không ghi âm thanh lên Google Drive; chỉ checkpoint, model và kết quả đánh giá (văn bản) được lưu trên Drive.
