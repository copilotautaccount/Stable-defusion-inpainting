# Stable Diffusion Inpainting

Dự án này cho phép bạn **fine-tune** (tinh chỉnh) và **chạy inference** với mô hình Stable Diffusion Inpainting trên bộ dữ liệu riêng của mình.

---

## Mục lục

- [Yêu cầu](#yêu-cầu)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [File `inference.py` — Giải thích chi tiết](#file-inferencepy--giải-thích-chi-tiết)
- [Chuẩn bị dữ liệu](#chuẩn-bị-dữ-liệu)
- [Huấn luyện (train.py)](#huấn-luyện-trainpy)
- [Chạy Inference](#chạy-inference)
- [Lỗi thường gặp](#lỗi-thường-gặp)

---

## Yêu cầu

```bash
pip install -r requirements.txt
```

Cần GPU NVIDIA có CUDA (khuyến nghị ≥ 16 GB VRAM để huấn luyện ở độ phân giải 512×512).  
Inference có thể chạy với GPU 6 GB VRAM hoặc CPU (chậm hơn).

---

## Cấu trúc dự án

```
.
├── dataset.py       # PyTorch Dataset đọc ảnh và mask cho quá trình huấn luyện
├── train.py         # Script fine-tune mô hình Stable Diffusion Inpainting
├── inference.py     # Script chạy inference (tạo ảnh inpainting)
├── requirements.txt # Các thư viện Python cần thiết
└── README.md
```

---

## File `inference.py` — Giải thích chi tiết

File `inference.py` là script dùng để **chạy mô hình tạo ảnh inpainting**, hỗ trợ cả mô hình base lẫn mô hình đã được fine-tune.

### Các tính năng chính

| Tính năng | Mô tả |
|---|---|
| **Single-image mode** | Xử lý một cặp ảnh + mask duy nhất |
| **Batch directory mode** | Xử lý toàn bộ thư mục ảnh và mask cùng lúc |
| **LoRA weights** | Hỗ trợ tải thêm LoRA adapter để cải thiện kết quả |
| **Fine-tuned model** | Tải model đã được fine-tune bởi `train.py` |
| **Base model** | Dùng trực tiếp model `runwayml/stable-diffusion-inpainting` mà không cần fine-tune |

### Cấu trúc hàm trong `inference.py`

| Hàm | Mô tả |
|---|---|
| `load_pipeline(model_path, device, dtype, lora_weights)` | Tải pipeline Stable Diffusion Inpainting từ thư mục local hoặc Hugging Face. Tự động bật `attention_slicing` để tiết kiệm VRAM. |
| `prepare_inputs(image_path, mask_path, image_size)` | Tải và resize ảnh + mask. Mask được nhị phân hoá: pixel > 127 = vùng cần inpaint. |
| `collect_image_mask_pairs(image_dir, mask_dir)` | Ghép cặp ảnh và mask theo **tên file** (stem). Cảnh báo nếu có ảnh thiếu mask tương ứng. |
| `run_inference(pipe, image, mask, ...)` | Chạy pipeline trên một cặp ảnh/mask và trả về danh sách ảnh kết quả. |
| `parse_args()` | Phân tích toàn bộ tham số dòng lệnh. |
| `main()` | Điều phối toàn bộ luồng: validate tham số → tải model → xử lý ảnh → lưu kết quả. |

### Quy ước ảnh mask

| Màu pixel | Ý nghĩa |
|---|---|
| **Trắng (255)** | Vùng cần inpaint — mô hình sẽ tạo nội dung mới tại đây |
| **Đen (0)** | Vùng giữ nguyên — mô hình không thay đổi vùng này |

---

## Chuẩn bị dữ liệu

Tổ chức dữ liệu theo cấu trúc sau:

```
data/
├── images/          # Ảnh RGB gốc  (*.png / *.jpg)
├── masks/           # Ảnh mask nhị phân  (*.png / *.jpg)
│                    #   Trắng (255) = vùng cần fill
│                    #   Đen   (0)  = vùng giữ nguyên
└── prompts.txt      # (tuỳ chọn) Mỗi dòng là một text prompt tương ứng với ảnh
```

- Ảnh và mask được **ghép cặp theo thứ tự sort** (khi dùng `train.py`).
- Khi dùng `inference.py` chế độ batch, ảnh và mask được ghép theo **tên file** (stem).
- Nếu thiếu `prompts.txt`, toàn bộ ảnh sẽ được huấn luyện với prompt rỗng.

---

## Huấn luyện (`train.py`)

```bash
python train.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-inpainting \
    --data_dir ./data \
    --output_dir ./output \
    --num_train_epochs 10 \
    --train_batch_size 2 \
    --learning_rate 1e-5 \
    --image_size 512 \
    --save_steps 500 \
    --mixed_precision fp16
```

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--pretrained_model_name_or_path` | `runwayml/stable-diffusion-inpainting` | Model base (HF id hoặc đường dẫn local) |
| `--data_dir` | *(bắt buộc)* | Thư mục gốc của bộ dữ liệu |
| `--output_dir` | `./output` | Nơi lưu checkpoints và model cuối |
| `--image_size` | `512` | Độ phân giải huấn luyện |
| `--train_batch_size` | `2` | Batch size trên mỗi GPU |
| `--num_train_epochs` | `10` | Số epoch huấn luyện |
| `--learning_rate` | `1e-5` | Learning rate ban đầu |
| `--save_steps` | `500` | Lưu checkpoint mỗi N bước |
| `--mixed_precision` | `no` | `fp16` / `bf16` / `no` |
| `--gradient_accumulation_steps` | `1` | Tích luỹ gradient trước khi update |
| `--use_8bit_adam` | `False` | Bật 8-bit Adam (cần `bitsandbytes`) |

Sau khi huấn luyện, toàn bộ pipeline (VAE, UNet đã fine-tune, text encoder, tokenizer) sẽ được lưu vào `--output_dir` theo định dạng Diffusers và có thể load trực tiếp cho inference.

---

## Chạy Inference

### 1. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 2. Chế độ một ảnh (Single-image)

Dùng model đã fine-tune:

```bash
python inference.py \
    --model_path ./output \
    --image ./data/images/photo.png \
    --mask  ./data/masks/photo.png \
    --prompt "a beautiful garden" \
    --output_dir ./results
```

Dùng thẳng model base (không cần fine-tune):

```bash
python inference.py \
    --model_path runwayml/stable-diffusion-inpainting \
    --image ./photo.png \
    --mask  ./mask.png \
    --prompt "a cozy living room"
```

### 3. Chế độ batch (nhiều ảnh cùng lúc)

```bash
python inference.py \
    --model_path ./output \
    --image_dir ./data/images \
    --mask_dir  ./data/masks \
    --prompt "a beautiful garden" \
    --output_dir ./results
```

Ảnh và mask được ghép cặp theo **tên file** (ví dụ `photo.png` ↔ `photo.png`).

### 4. Nâng cao — với LoRA và tuỳ chỉnh đầy đủ

```bash
python inference.py \
    --model_path     ./output \
    --lora_weights   ./lora_adapter \
    --image          ./photo.png \
    --mask           ./mask.png \
    --prompt         "a modern kitchen with marble countertops" \
    --negative_prompt "blurry, low quality, distorted" \
    --num_inference_steps 75 \
    --guidance_scale 9.0 \
    --strength       0.9 \
    --num_images_per_prompt 3 \
    --seed           1234 \
    --output_dir     ./results \
    --image_size     512 \
    --mixed_precision fp16
```

### 5. Danh sách đầy đủ tham số `inference.py`

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--model_path` | `./output` | Thư mục model fine-tune hoặc HF model id |
| `--lora_weights` | `None` | Thư mục chứa LoRA adapter (tuỳ chọn) |
| `--image` | `None` | Ảnh gốc (chế độ single-image) |
| `--mask` | `None` | Ảnh mask (chế độ single-image) |
| `--image_dir` | `None` | Thư mục ảnh (chế độ batch) |
| `--mask_dir` | `None` | Thư mục mask (chế độ batch) |
| `--prompt` | `""` | Text prompt mô tả nội dung cần tạo |
| `--negative_prompt` | `"low quality, blurry, distorted"` | Negative prompt |
| `--num_inference_steps` | `50` | Số bước khử nhiễu |
| `--guidance_scale` | `7.5` | Mức độ tuân theo prompt (CFG scale) |
| `--strength` | `1.0` | Mức độ biến đổi vùng mask (0–1) |
| `--num_images_per_prompt` | `1` | Số ảnh tạo ra cho mỗi đầu vào |
| `--seed` | `None` | Seed ngẫu nhiên để tái tạo kết quả |
| `--output_dir` | `./results` | Thư mục lưu ảnh kết quả |
| `--image_size` | `512` | Độ phân giải inference (hình vuông) |
| `--device` | tự động | `cuda` / `cpu` |
| `--mixed_precision` | `fp16` | `fp16` / `bf16` / `no` |

### 6. Xem trợ giúp

```bash
python inference.py --help
```

---

## Lỗi thường gặp

| Lỗi | Nguyên nhân | Giải pháp |
|---|---|---|
| `CUDA out of memory` | VRAM không đủ | Dùng `--mixed_precision fp16` hoặc giảm `--image_size` |
| `ValueError: Provide either --image + --mask ...` | Không truyền đủ tham số đầu vào | Truyền `--image` + `--mask` hoặc `--image_dir` + `--mask_dir` |
| `FileNotFoundError: No matching image/mask pairs` | Tên file ảnh và mask không khớp nhau | Đặt tên file ảnh và mask giống nhau (chỉ khác extension) |
| `OSError: model not found` | Chưa có kết nối internet để tải model | Kết nối mạng và chạy lại lần đầu để tải cache |
| Ảnh kết quả bị mờ | Số bước quá ít | Tăng `--num_inference_steps` lên 75–100 |
| Kết quả không theo prompt | `guidance_scale` chưa phù hợp | Thử `--guidance_scale` từ 7 đến 12 |
