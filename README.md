# Stable Diffusion Inpainting

Dự án này sử dụng mô hình **Stable Diffusion Inpainting** để tự động điền (inpaint) vùng bị che trong ảnh dựa trên mô tả văn bản.

---

## 📄 File `inference.py` — Giải thích chi tiết

File `inference.py` là tập lệnh suy luận (inference) chính. Nó thực hiện toàn bộ quy trình từ tải mô hình đến xuất ảnh kết quả.

### Cấu trúc file

| Thành phần | Mô tả |
|---|---|
| `load_image(path)` | Tải ảnh đầu vào từ đường dẫn và chuyển sang định dạng RGB |
| `load_mask(path)` | Tải ảnh mặt nạ (mask) và chuyển sang ảnh xám (greyscale) |
| `get_device()` | Tự động chọn thiết bị: GPU (`cuda`) nếu có, ngược lại dùng CPU |
| `load_pipeline(model_id, device)` | Tải pipeline Stable Diffusion Inpainting từ Hugging Face (hoặc từ cache) |
| `run_inpainting(...)` | Hàm chính thực hiện quá trình inpainting: resize ảnh, chạy mô hình, trả về ảnh kết quả |
| `parse_args()` | Phân tích tham số dòng lệnh (CLI arguments) |
| `main()` | Hàm điều phối toàn bộ luồng xử lý |

### Quy ước ảnh mặt nạ (mask)

| Màu pixel | Ý nghĩa |
|---|---|
| **Trắng (255)** | Vùng cần inpaint — mô hình sẽ tạo nội dung mới tại đây |
| **Đen (0)** | Vùng giữ nguyên — mô hình không thay đổi vùng này |

### Các tham số chính của `run_inpainting`

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `prompt` | *(bắt buộc)* | Mô tả văn bản về nội dung cần tạo trong vùng mask |
| `negative_prompt` | `""` | Mô tả những gì **không** muốn xuất hiện trong ảnh |
| `width` / `height` | `512` | Kích thước ảnh đầu ra (phải là bội số của 8) |
| `num_inference_steps` | `50` | Số bước khử nhiễu — nhiều hơn → chất lượng cao hơn nhưng chậm hơn |
| `guidance_scale` | `7.5` | Mức độ tuân theo prompt (thường từ 5 đến 15) |
| `seed` | `42` | Seed ngẫu nhiên để tái tạo kết quả |

---

## 🚀 Hướng dẫn chạy file `inference.py`

### 1. Yêu cầu hệ thống

- Python 3.8 trở lên
- GPU NVIDIA với CUDA (khuyến nghị, ít nhất 6 GB VRAM) hoặc CPU (chậm hơn)

### 2. Cài đặt thư viện

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate Pillow
```

> **Lưu ý:** Nếu chỉ dùng CPU, cài torch bình thường:
> ```bash
> pip install torch torchvision
> ```

### 3. Chuẩn bị file đầu vào

Bạn cần hai file ảnh:

- **`input.png`** — Ảnh gốc mà bạn muốn chỉnh sửa.
- **`mask.png`** — Ảnh mặt nạ cùng kích thước với ảnh gốc:
  - Tô **trắng** (`255`) vào vùng muốn inpaint.
  - Giữ **đen** (`0`) cho vùng muốn giữ nguyên.

### 4. Chạy lệnh cơ bản

```bash
python inference.py \
    --image input.png \
    --mask  mask.png  \
    --prompt "a beautiful garden with flowers"
```

Kết quả sẽ được lưu tại `output.png` (mặc định).

### 5. Tùy chỉnh nâng cao

```bash
python inference.py \
    --image       input.png \
    --mask        mask.png  \
    --prompt      "a modern kitchen with marble countertops" \
    --negative-prompt "blurry, low quality, distorted" \
    --output      result.png \
    --width       512 \
    --height      512 \
    --steps       75 \
    --guidance-scale 9.0 \
    --seed        1234
```

### 6. Danh sách đầy đủ các tham số CLI

| Tham số | Bắt buộc | Mặc định | Mô tả |
|---|---|---|---|
| `--image` | ✅ | — | Đường dẫn ảnh đầu vào |
| `--mask` | ✅ | — | Đường dẫn ảnh mặt nạ |
| `--prompt` | ✅ | — | Mô tả văn bản cho vùng cần tạo |
| `--output` | | `output.png` | Đường dẫn file kết quả |
| `--model` | | `runwayml/stable-diffusion-inpainting` | ID mô hình trên Hugging Face |
| `--negative-prompt` | | `""` | Mô tả nội dung không muốn xuất hiện |
| `--width` | | `512` | Chiều rộng ảnh đầu ra |
| `--height` | | `512` | Chiều cao ảnh đầu ra |
| `--steps` | | `50` | Số bước khử nhiễu |
| `--guidance-scale` | | `7.5` | Mức độ tuân theo prompt |
| `--seed` | | `42` | Seed ngẫu nhiên |

### 7. Xem trợ giúp

```bash
python inference.py --help
```

---

## 🔄 Luồng xử lý tổng quát

```
Ảnh gốc (input.png)  ─┐
                        ├──► run_inpainting() ──► output.png
Ảnh mask (mask.png)  ─┘         ▲
                                 │
                         Mô hình SD Inpainting
                         + Text Prompt
```

---

## ❓ Lỗi thường gặp

| Lỗi | Nguyên nhân | Giải pháp |
|---|---|---|
| `CUDA out of memory` | VRAM không đủ | Thêm `--width 512 --height 512` hoặc dùng CPU |
| `OSError: ... not found` | Chưa có kết nối internet để tải mô hình | Kết nối mạng và chạy lại lần đầu để tải cache |
| Ảnh đầu ra bị mờ | Số bước quá ít | Tăng `--steps` lên 75–100 |
| Kết quả không theo ý | `guidance_scale` chưa phù hợp | Thử `--guidance-scale` từ 7 đến 12 |
