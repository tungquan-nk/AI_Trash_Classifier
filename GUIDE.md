Dưới đây là hướng dẫn dành cho thành viên trong nhóm khi pull project AI phân loại rác từ Git về máy tính và chạy ứng dụng Streamlit.

Sếp có thể gửi nguyên nội dung này cho các thành viên trong nhóm để mọi người làm theo.

# HƯỚNG DẪN CÀI ĐẶT VÀ CHẠY PROJECT AI PHÂN LOẠI RÁC

## 1. Chuẩn bị môi trường

Trước khi tải project, cần cài đặt:

| Công cụ                     | Mục đích                   |
| --------------------------- | -------------------------- |
| Git                         | Tải mã nguồn từ GitHub     |
| Python 3.10 trở lên         | Chạy chương trình Python   |
| VS Code                     | Môi trường lập trình       |
| Ubuntu/WSL hoặc Terminal    | Thực hiện các lệnh cài đặt |
| NVIDIA GPU (không bắt buộc) | Tăng tốc quá trình dự đoán |

Nếu sử dụng Windows, có thể chạy trực tiếp bằng PowerShell hoặc sử dụng Ubuntu thông qua WSL.

## 2. Pull project từ GitHub

Mở Terminal và di chuyển đến thư mục muốn lưu project.

Ví dụ:

```
cd ~/PRJ3
```

Clone repository:

```
git clone <GIT_REPOSITORY_URL>
```

Thay `<GIT_REPOSITORY_URL>` bằng đường dẫn GitHub thực tế của nhóm.

Di chuyển vào thư mục dự án:

```
cd AI_Waste_Classification
```

Nếu máy đã có project và muốn cập nhật code mới nhất:

```
git pull origin main
```

Lưu ý: Nếu nhóm sử dụng branch khác `main`, hãy thay bằng tên branch tương ứng.

## 3. Tạo môi trường ảo Python

Mỗi thành viên cần tạo môi trường ảo riêng trên máy của mình.

Tạo môi trường ảo:

```
python3 -m venv .venv
```

Kích hoạt môi trường ảo trên Ubuntu/WSL:

```
source .venv/bin/activate
```

Trên Windows PowerShell:

```
.venv\Scripts\Activate.ps1
```

Sau khi kích hoạt thành công, Terminal sẽ hiển thị tiền tố `(.venv)`.

Ví dụ:

```
(.venv) phuanh@LAP-PB:~/PRJ3/AI_Waste_Classification$
```

## 4. Cài đặt các thư viện cần thiết

### 4.1. Cập nhật pip

```
python -m pip install --upgrade pip
```

### 4.2. Cài đặt PyTorch và Torchvision

Nếu sử dụng Ubuntu/WSL và muốn tận dụng GPU NVIDIA hỗ trợ CUDA 12.6:

```
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Nếu máy không có GPU NVIDIA hoặc chỉ cần chạy ứng dụng bằng CPU:

```
python -m pip install torch torchvision
```

### 4.3. Cài đặt các thư viện còn lại

```
python -m pip install streamlit pillow numpy pandas matplotlib scikit-learn
```

Các thư viện có tác dụng:

| Thư viện     | Chức năng                        |
| ------------ | -------------------------------- |
| torch        | Xây dựng và chạy mô hình học sâu |
| torchvision  | Cung cấp ResNet18 và xử lý ảnh   |
| streamlit    | Xây dựng giao diện web           |
| pillow       | Đọc và xử lý hình ảnh            |
| numpy        | Xử lý dữ liệu dạng mảng          |
| pandas       | Đọc và xử lý dữ liệu             |
| matplotlib   | Vẽ biểu đồ, ma trận nhầm lẫn     |
| scikit-learn | Đánh giá mô hình bằng các chỉ số |

## 5. Kiểm tra môi trường

Kiểm tra phiên bản Python:

```
python --version
```

Kiểm tra PyTorch:

```
python -c "import torch; print('PyTorch:', torch.__version__)"
```

Kiểm tra GPU:

```
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Nếu kết quả là `True`, PyTorch có thể sử dụng CUDA trên máy hiện tại.

Nếu kết quả là `False`, ứng dụng vẫn có thể chạy bằng CPU.

## 6. Kiểm tra các file mô hình

Sau khi pull code, cần đảm bảo repository có checkpoint và danh sách lớp của ResNet18 V2.

Cấu trúc thư mục quan trọng:

```
AI_Waste_Classification/
│
├── app.py
│
├── checkpoints/
│   └── resnet18_v2_best.pth
│
├── artifacts/
│   └── resnet18_v2_classes.json
│
├── src/
│   ├── train_resnet.py
│   ├── evaluate_resnet_v2.py
│   ├── merge_datasets.py
│   └── split_merged_dataset.py
│
├── datasets/
│
└── .venv/
```

Lưu ý quan trọng:

- `.venv/` không cần và không nên đưa lên GitHub. Mỗi thành viên tự tạo môi trường ảo.
- `checkpoints/resnet18_v2_best.pth` là file trọng số mô hình đã huấn luyện.
- `artifacts/resnet18_v2_classes.json` chứa thứ tự các nhãn phân loại.
- Nếu checkpoint không được đưa lên GitHub, thành viên cần nhận file này từ người quản lý mô hình hoặc tải từ nơi nhóm lưu trữ.
- Không cần tải dataset nếu chỉ muốn chạy ứng dụng dự đoán, miễn là checkpoint và file classes đã có.

Kiểm tra hai file mô hình:

```
ls -lh checkpoints/resnet18_v2_best.pth
ls -lh artifacts/resnet18_v2_classes.json
```

## 7. Chạy ứng dụng Streamlit

Sau khi cài đặt đầy đủ thư viện và kiểm tra file mô hình, chạy:

```
python -m streamlit run app.py
```

Terminal sẽ hiển thị địa chỉ:

```
Local URL: http://localhost:8501
```

Mở trình duyệt và truy cập:

[http://localhost:8501](http://localhost:8501?utm_source=chatgpt.com)

### Quy trình sử dụng

1. Mở giao diện ứng dụng.
2. Nhấn nút tải ảnh lên.
3. Chọn ảnh rác từ máy tính.
4. Hệ thống xử lý ảnh và đưa vào mô hình ResNet18 V2.
5. Mô hình dự đoán loại rác.
6. Giao diện hiển thị kết quả phân loại.

Các lớp rác có thể nhận diện:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

## 8. Một số lỗi thường gặp

Lỗi 1: `ModuleNotFoundError: No module named 'torch'`

Nguyên nhân: Chưa cài PyTorch trong môi trường ảo đang sử dụng.

Cách khắc phục:

```
source .venv/bin/activate
python -m pip install torch torchvision
```

Lỗi 2: `ModuleNotFoundError: No module named 'streamlit'`

Cách khắc phục:

```
python -m pip install streamlit
```

Lỗi 3: Không tìm thấy checkpoint

Ví dụ:

```
FileNotFoundError: checkpoints/resnet18_v2_best.pth
```

Nguyên nhân: File trọng số mô hình chưa có trên máy hoặc đường dẫn trong code không chính xác.

Cách khắc phục: Kiểm tra thư mục `checkpoints/` và nhận file checkpoint từ thành viên phụ trách AI.

Lỗi 4: `CUDA available: False`

Không nhất thiết là lỗi. Nếu máy không có GPU NVIDIA tương thích hoặc môi trường chưa nhận GPU, PyTorch vẫn có thể chạy trên CPU.

Lỗi 5: Git pull bị xung đột

Nếu có chỉnh sửa code chưa commit, Git có thể không cho pull trực tiếp. Hãy commit hoặc lưu lại thay đổi trước khi cập nhật.

## 9. Các lệnh thường dùng

| Công việc            | Lệnh                             |
| -------------------- | -------------------------------- |
| Tải project lần đầu  | `git clone <URL>`                |
| Cập nhật code        | `git pull origin main`           |
| Tạo môi trường ảo    | `python3 -m venv .venv`          |
| Kích hoạt môi trường | `source .venv/bin/activate`      |
| Cài thư viện         | `python -m pip install ...`      |
| Chạy ứng dụng        | `python -m streamlit run app.py` |
| Dừng ứng dụng        | `Ctrl + C`                       |
| Thoát môi trường ảo  | `deactivate`                     |

## 10. Kết quả mong đợi

Sau khi hoàn thành tất cả các bước, mỗi thành viên có thể chạy ứng dụng AI phân loại rác trên máy tính cá nhân mà không cần huấn luyện lại mô hình.

Mô hình sử dụng: ResNet18 V2

Kết quả đánh giá hiện tại trên tập Test:

- Accuracy: 92.26%
- Macro-F1: 92.14%
- Weighted-F1: 92.22%
- Số lượng lớp phân loại: 6

## Một việc nhóm nên làm để tránh lỗi khi cài đặt

Tôi khuyên sếp bổ sung file `requirements.txt` vào repository để thành viên không phải tự cài từng thư viện.

Có thể tạo file bằng lệnh:

```
cat > requirements.txt <<'EOF'
streamlit
pillow
numpy
pandas
matplotlib
scikit-learn
EOF
```

Sau đó mọi người chỉ cần chạy:

```
python -m pip install -r requirements.txt
```

PyTorch nên được cài riêng theo CPU hoặc CUDA để tránh cài nhầm phiên bản.

Lưu ý quan trọng nhất: Nếu nhóm muốn mọi thành viên chỉ cần `git clone` là chạy được ngay, cần đảm bảo file `resnet18_v2_best.pth` và `resnet18_v2_classes.json` được chia sẻ qua Git LFS hoặc một kho lưu trữ chung. Không nên đưa toàn bộ dataset và môi trường `.venv` lên GitHub.