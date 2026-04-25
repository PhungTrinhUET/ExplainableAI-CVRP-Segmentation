# Kế hoạch thực nghiệm CVRP — Tái tạo & Mở rộng bài báo

## Tình trạng dữ liệu (đã xác nhận)

| Thành phần | Số lượng | Kích thước | Trạng thái |
|---|---|---|---|
| Multi-cultivar images | 2205 ảnh, 282 giống (50 G + 232 T) | 1080×1920 | ✅ Sẵn sàng |
| Multi-cultivar annotations | 2205 mask | 1080×1920, vals=[0,1] | ✅ Sẵn sàng |
| Field scene images | 100 ảnh | ~800×600 (biến đổi) | ✅ Sẵn sàng |
| Field scene annotations | 100 mask | tương ứng, vals=[0,1] | ✅ Đã convert |
| Indoor panicle images | 123 ảnh, 41 giống, 3 ảnh/giống | 1500×2007 | ✅ Sẵn sàng |
| 3D .ply files | 3 file (G1204, G1210, G1224) | — | ✅ Có sẵn |
| Data split (train/val) | 2116 train / 80 val / 100 field_test | — | ✅ Đã chia |

---

## Cấu trúc thư mục dự án

```
CVRP_xAI/
├── CVRP_Rice/CVRP/              ← Raw dataset gốc (KHÔNG sửa)
├── data/CVRP/                   ← Dữ liệu đã chuẩn bị cho MMSeg
│   ├── img_dir/{train,val,field_test}/
│   └── ann_dir/{train,val,field_test}/
├── experiments/
│   ├── segmentation/
│   │   ├── configs/             ← Config files cho 4 model
│   │   └── work_dirs/           ← Checkpoint + log training
│   ├── sam_grains/              ← Kết quả SAM/SAM2
│   └── reconstruction/          ← Render 3D từ .ply
├── results/
│   ├── tables/                  ← CSV kết quả số (metrics)
│   └── figures/
│       ├── 01_dataset_analysis/ ← Phân tích dữ liệu
│       ├── 02_segmentation_qualitative/ ← Ảnh kết quả phân đoạn
│       ├── 03_segmentation_curves/      ← Loss/metric curves
│       ├── 04_sam_grains/       ← Kết quả đếm hạt
│       └── 05_reconstruction/   ← Ảnh render 3D
├── checkpoints/                 ← Pretrained weights tải về
├── scripts/                     ← Tất cả script Python
└── report/                      ← File báo cáo LaTeX
    └── figures/                 ← Symlink từ results/figures
```

---

## Kế hoạch kết quả cần có

### PHẦN 1 — Phân tích Dataset (không cần GPU)

**Mục tiêu:** Hiểu và trình bày đặc điểm dataset trước khi thực nghiệm

#### 1.1 — Bảng thống kê dataset
- **File output:** `results/tables/dataset_statistics.csv`
- **Nội dung:** Đúng Table 1 trong paper + thêm phân tích chi tiết:
  ```
  | Loại        | Số giống | Số ảnh | Số annotation | Kích thước ảnh |
  |-------------|----------|--------|---------------|----------------|
  | Multi-cult. | 282      | 2205   | 2205          | 1080×1920      |
  | Field scene | —        | 100    | 100           | biến đổi       |
  | Indoor      | 41       | 123    | 0             | 1500×2007      |
  ```

#### 1.2 — Phân bố số ảnh mỗi giống
- **File output:** `results/figures/01_dataset_analysis/fig_images_per_cultivar.png`
- **Dạng:** Histogram (cột) — trục X: số ảnh/giống, trục Y: số giống có count đó
- **Vì sao cần:** Cho thấy dataset cân bằng hay lệch

#### 1.3 — Ảnh mẫu đa góc nhìn (như Figure 3 trong paper)
- **File output:** `results/figures/01_dataset_analysis/fig_multiview_sample.png`
- **Nội dung:** Grid 4 góc nhìn (nadir, oblique, side, close-up) của cùng 1 giống
- **Vì sao cần:** Minh họa sự đa dạng góc nhìn trong dataset

#### 1.4 — Annotation overlay (ảnh + mask chồng lên)
- **File output:** `results/figures/01_dataset_analysis/fig_annotation_examples.png`
- **Nội dung:** Grid 6 cặp (ảnh gốc | ảnh + mask đỏ) từ các giống khác nhau
- **Vì sao cần:** Cho thấy chất lượng annotation

#### 1.5 — Phân tích annotation (tỉ lệ pixel bông/nền)
- **File output:** `results/tables/annotation_ratio.csv`
- **Nội dung:** Tỉ lệ % pixel bông trung bình, min, max — phân tách train/val
- **Vì sao cần:** Cho thấy imbalance giữa foreground/background

---

### PHẦN 2 — Semantic Segmentation (cần GPU)

**Mục tiêu:** Tái tạo Table 2 và Table 3 trong paper, cộng thêm visualizations

#### 2.1 — Training 4 model
- **Model:** DeepLabV3+ (ResNet), Mask2Former (Swin-B), SegFormer (MiT-B2), K-Net (Swin-B)
- **Config:** batch=2, optimizer=AdamW, lr=0.0001, 80k iterations
- **Output:** `experiments/segmentation/work_dirs/{model_name}/`
  - `best_mIoU_iter_*.pth` — checkpoint tốt nhất
  - `{timestamp}.log` — log training

#### 2.2 — Bảng kết quả số (tái tạo Table 2 + Table 3)
- **File output:** `results/tables/seg_multicult_results.csv`
- **File output:** `results/tables/seg_fieldscene_results.csv`
- **Nội dung:**
  ```
  | Model       | Precision | Recall | F-score | IoU   |
  |-------------|-----------|--------|---------|-------|
  | DeepLabV3+  | ??.??     | ??.??  | ??.??   | ??.?? |
  | Mask2Former | ??.??     | ??.??  | ??.??   | ??.?? |
  | SegFormer   | ??.??     | ??.??  | ??.??   | ??.?? |
  | K-Net       | ??.??     | ??.??  | ??.??   | ??.?? |
  ```
  → So sánh với số trong paper, tính % deviation

#### 2.3 — Training curves (Loss + IoU theo iteration)
- **File output:** `results/figures/03_segmentation_curves/fig_training_curves.png`
- **Nội dung:** 2 subplots:
  - Trái: Train Loss theo iteration (4 model cùng 1 biểu đồ)
  - Phải: Val IoU theo iteration (4 model cùng 1 biểu đồ)
- **Vì sao cần:** Cho thấy convergence behavior và overfitting nếu có

#### 2.4 — Visualization kết quả phân đoạn (như Figure 6 + 7 trong paper)
- **File output:** `results/figures/02_segmentation_qualitative/`
  - `fig_success_cases.png` — 4 ảnh phân đoạn tốt
  - `fig_failure_weed.png` — cỏ dại bị nhầm là bông
  - `fig_failure_leaf.png` — lá bị nhầm là bông
  - `fig_failure_small.png` — bông nhỏ bị bỏ sót
  - `fig_4models_comparison.png` — Grid: 4 model × 4 ảnh (như Figure 7)
- **Format:** Ảnh gốc + mask overlay màu cho từng model (màu: xanh=TP, đỏ=FP, vàng=FN)

#### 2.5 — Biểu đồ radar (tùy chọn nhưng đẹp cho báo cáo)
- **File output:** `results/figures/03_segmentation_curves/fig_radar_comparison.png`
- **Nội dung:** Radar chart 4 trục (Precision/Recall/F-score/IoU) cho 4 model
- **Vì sao cần:** So sánh toàn diện 4 model trong 1 hình

---

### PHẦN 3 — SAM/SAM2 Grain Segmentation (cần GPU nhẹ hơn)

**Mục tiêu:** Tái tạo kết quả Section 3.3 — đếm hạt trên ảnh trong phòng

#### 3.1 — Kết quả đếm hạt
- **File output:** `results/tables/sam_grain_results.csv`
- **Nội dung:**
  ```
  | Model | Trạng thái   | % hạt phát hiện | Avg/giống |
  |-------|-------------|-----------------|-----------|
  | SAM   | Natural     | ~73%            | ??        |
  | SAM   | Unfolded    | ~83%            | ??        |
  | SAM2  | Natural     | ??%             | ??        |
  | SAM2  | Unfolded    | ??%             | ??        |
  ```
  → Paper chỉ có SAM. SAM2 là phần mở rộng thêm.

#### 3.2 — Visualization phân đoạn hạt
- **File output:** `results/figures/04_sam_grains/`
  - `fig_sam_natural.png` — 3 bông trạng thái tự nhiên + mask SAM
  - `fig_sam_unfolded.png` — 3 bông trạng thái mở + mask SAM
  - `fig_sam_vs_sam2.png` — So sánh SAM và SAM2 trên cùng bông
- **Vì sao cần:** Cho thấy tại sao unfolded dễ hơn, visualize thách thức occlusion

---

### PHẦN 4 — 3D Reconstruction (không cần training)

**Mục tiêu:** Visualize 3 file .ply đã có sẵn

#### 4.1 — Render 3D views
- **File output:** `results/figures/05_reconstruction/`
  - `fig_G1204_plain.png` — render 3D không annotation
  - `fig_G1204_annotated.png` — render 3D với bông highlight vàng
  - `fig_G1210_plain.png`, `fig_G1210_annotated.png`
  - `fig_G1224_plain.png`, `fig_G1224_annotated.png`
- **Tool:** Open3D (nhẹ, dễ dùng) hoặc PyVista
- **Vì sao cần:** Tái tạo Figure 8 trong paper

---

## Thứ tự thực hiện

```
Giai đoạn 1 — Không cần GPU (làm ngay)
├── [1.1] Thống kê dataset
├── [1.2] Histogram ảnh/giống
├── [1.3] Multiview sample grid
├── [1.4] Annotation overlay grid
└── [1.5] Tỉ lệ annotation

Giai đoạn 2 — Cần GPU (song song với Giai đoạn 1)
├── [2.1] Cài môi trường + MMSegmentation
├── [2.1] Train 4 model (lần lượt hoặc song song)
├── [2.2] Evaluate → tables
├── [2.3] Export training curves từ log
└── [2.4] Visualize kết quả phân đoạn

Giai đoạn 3 — Sau khi có model tốt nhất
├── [3.1] Chạy SAM/SAM2 trên IndoorPanicleImages
└── [3.2] Visualize grain segmentation

Giai đoạn 4 — Cuối cùng
├── [4.1] Render 3D từ .ply
└── Tổng hợp vào báo cáo LaTeX
```

---

## Checklist kết quả

### Bảng số (Tables)
- [ ] `dataset_statistics.csv`
- [ ] `annotation_ratio.csv`
- [ ] `seg_multicult_results.csv` — tái tạo Table 2
- [ ] `seg_fieldscene_results.csv` — tái tạo Table 3
- [ ] `sam_grain_results.csv`

### Hình ảnh (Figures)
- [ ] `fig_images_per_cultivar.png`
- [ ] `fig_multiview_sample.png`
- [ ] `fig_annotation_examples.png`
- [ ] `fig_training_curves.png`
- [ ] `fig_success_cases.png`
- [ ] `fig_failure_cases.png` (3 loại lỗi)
- [ ] `fig_4models_comparison.png`
- [ ] `fig_radar_comparison.png`
- [ ] `fig_sam_natural.png` + `fig_sam_unfolded.png`
- [ ] `fig_sam_vs_sam2.png`
- [ ] `fig_G120x_plain.png` + `fig_G120x_annotated.png` (×3 giống)