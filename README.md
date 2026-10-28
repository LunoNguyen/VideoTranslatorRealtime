# VideoTranslatorRealtime

Ứng dụng Windows thu âm thanh của các ứng dụng bạn cho phép, nhận dạng giọng nói (tiếng Anh / tiếng Nhật), dịch sang tiếng Việt và hiển thị phụ đề nổi trên màn hình, có thể đọc bản dịch bằng giọng nói.

## Yêu cầu phần cứng

- Hệ điều hành: Windows 10 phiên bản 2004 trở lên, hoặc Windows 11 (cần cho tính năng thu âm theo từng ứng dụng).
- GPU: NVIDIA có từ 4 GB VRAM trở lên (khuyến nghị). GPU 6 GB VRAM trở lên có thể dùng preset `6gb` (Whisper `medium`) để nhận dạng chính xác hơn.
- RAM: từ 8 GB trở lên.
- Không có GPU vẫn chạy được trên CPU nhưng chậm hơn rõ rệt.

## Giấy phép

- Mã nguồn: giấy phép MIT, xem file `LICENSE`.
- Các model và dữ liệu đi kèm có giấy phép riêng:

| Thành phần | Giấy phép |
|---|---|
| Whisper (qua faster-whisper) | MIT |
| NLLB-200-distilled-600M (bản CTranslate2 int8) | CC-BY-NC 4.0, **không dùng cho mục đích thương mại** |
| Giọng đọc Piper `vi_VN-vais1000-medium` | CC BY 4.0 |
| Model phân biệt người nói WeSpeaker ResNet34 (VoxCeleb) | CC-BY-4.0 |

- Dữ liệu huấn luyện (chỉ dùng khi tự fine-tune): mỗi bộ dữ liệu có giấy phép riêng, xem chi tiết trong `training/data_sources.md`. Một số bộ chỉ cho phép phi thương mại hoặc chỉ dành cho nghiên cứu (ví dụ TED/IWSLT/TED2020 theo CC BY-NC-ND 4.0, ReazonSpeech theo Điều 30-4 Luật Bản quyền Nhật Bản, PhoMT chỉ dành cho nghiên cứu và mặc định tắt). Bạn cần đọc và tuân thủ điều khoản của từng bộ trước khi dùng hoặc phân phối model đã huấn luyện.

## Điều khoản sử dụng

- Bạn tự chịu trách nhiệm về quyền thu âm và quyền đối với nội dung được thu, đặc biệt với cuộc gọi hoặc cuộc họp có người khác tham gia: hãy bảo đảm bạn có quyền thu và xử lý âm thanh đó theo quy định pháp luật nơi bạn sống.
- Ứng dụng chỉ thu âm thanh của những ứng dụng bạn đã tích chọn "Cho phép nghe".
- Mặc định ứng dụng không lưu âm thanh hay bản ghi lời ra đĩa. Nhật ký (log) chỉ được ghi khi bạn tự bật.
- Không có telemetry. Ứng dụng chỉ kết nối mạng khi tải model; máy chủ nội bộ chỉ lắng nghe tại `127.0.0.1:8765`.
- Phần mềm được cung cấp "như hiện có", không kèm bảo hành (xem `LICENSE`). Bản dịch tự động có thể sai.
- Bạn phải tuân thủ giới hạn phi thương mại của các model và dữ liệu nêu ở mục Giấy phép.

## Cài đặt

1. Cài Python 3.11 trở lên (đã thử trên 3.12).
2. Tạo môi trường ảo và cài ứng dụng:

   ```
   py -3.12 -m venv .venv
   .venv\Scripts\activate
   pip install -e .
   ```

   Nếu có GPU NVIDIA, cài thêm các thư viện CUDA:

   ```
   pip install -e .[gpu]
   ```

3. Tải model gốc:

   ```
   python scripts/download_models.py --preset 4gb
   ```

   Dùng `--preset 6gb` để tải Whisper `medium`. Có thể tải riêng từng phần bằng `--only asr`, `--only mt`, `--only tts` hoặc `--only speaker`. Lệnh này cũng tải model phân biệt người nói (WeSpeaker ResNet34). Lệnh có thể chạy lại an toàn, phần đã tải sẽ được bỏ qua.

4. (Tùy chọn) Cài model đã fine-tune trên Google Colab.

   Huấn luyện trên Colab:
   - Mở notebook trong thư mục `training/` trên Colab (File, rồi Upload notebook; hoặc mở từ GitHub): `finetune_nllb_colab.ipynb` cho dịch en/ja sang vi, `finetune_whisper_ja_colab.ipynb` cho nhận dạng tiếng Nhật.
   - Chọn Runtime, rồi Change runtime type, rồi T4 GPU.
   - Chọn Run all, cho phép gắn Google Drive khi được hỏi. Với notebook Whisper, cần đăng nhập Hugging Face và chấp nhận điều khoản của bộ dữ liệu ReazonSpeech.
   - Khi chạy xong, tải file zip kết quả về máy.
   - Nếu huấn luyện qua nhiều phiên nối tiếp, hãy dọn Thùng rác của Google Drive giữa các phiên (Drive miễn phí chỉ có 15 GB).

   Cài model vào ứng dụng:
   - Model dịch: sao lưu rồi làm trống thư mục `models/mt/`, sau đó giải nén `nllb-vi-ct2-int8.zip` vào đó.
   - Model nhận dạng tiếng Nhật: giải nén `whisper-ja-ct2-int8.zip` vào `models/asr/small-ja/`, rồi thêm mục sau vào file `config.local.json` ở thư mục gốc (tạo file nếu chưa có; nếu file đã có thì giữ các mục khác):

     ```json
     {"asr": {"model": "small-ja"}}
     ```

     (Danh sách chọn model trong cửa sổ cài đặt chỉ có hai preset 4GB và 6GB nên model này được chọn qua file cấu hình; sau đó danh sách sẽ hiện thêm mục `small-ja (tùy chỉnh)`.)

## Sử dụng

1. Chạy ứng dụng:

   ```
   python run.py
   ```

   Cửa sổ cài đặt liệt kê các ứng dụng đang phát âm thanh.
2. Tích "Cho phép nghe" cho từng ứng dụng bạn muốn dịch.
3. Chọn ngôn ngữ nguồn: Auto, English hoặc 日本語.
4. Bật hoặc tắt đọc bản dịch bằng giọng nói (TTS) và chỉnh âm lượng.
5. Tích "Hiện câu gốc" nếu muốn hiển thị cả câu gốc cùng phụ đề.
6. Tích "Dịch tức thì" để phụ đề hiện tạm ngay khi đang nói rồi được thay bằng bản hoàn chỉnh; tích "Tô màu người nói" để mỗi người nói có một màu phụ đề riêng.
7. Phụ đề hiện trong cửa sổ nổi; kéo cửa sổ này để di chuyển đến vị trí mong muốn.
8. Đóng cửa sổ cài đặt sẽ dừng backend.

Thiết lập của bạn được lưu trong `config.local.json`. Giá trị mặc định nằm trong `config.default.json`.
