# Hướng dẫn sử dụng trên Overleaf

## Bước 1 — Upload files

Upload các file sau lên Overleaf:
- `main.tex` (file chính)
- `references.bib` (thư mục tài liệu tham khảo)

## Bước 2 — Upload hình ảnh

Tạo cấu trúc thư mục trên Overleaf như sau rồi upload từng nhóm:

```
figures/
├── 01_dataset_analysis/
│   ├── fig_annotation_examples.png
│   ├── fig_annotation_ratio_boxplot.png
│   ├── fig_images_per_cultivar.png
│   └── fig_multiview_sample.png
│
├── 02_segmentation_qualitative/
│   ├── fig_4models_grid.png
│   ├── fig_failure_cases.png
│   └── fig_success_cases.png
│
├── 03_segmentation_curves/
│   ├── fig_radar_comparison.png
│   ├── fig_results_comparison.png
│   └── fig_training_curves.png
│
├── 04_sam_grains/
│   ├── fig_sam_comparison.png
│   ├── fig_sam_natural.png
│   ├── fig_sam_per_cultivar.png
│   └── fig_sam_unfolded.png
│
├── 05_reconstruction/
│   └── fig_3d_reconstruction_grid.png
│
└── 06_xai/
    ├── comparison/fig_cnn_vs_transformer.png
    ├── counterfactual/fig_deletion_insertion.png
    ├── counterfactual/fig_occlusion_cf_map.png
    ├── counterfactual/fig_superpixel_cf.png
    ├── failure/fig_failure_cam.png
    ├── gradcam/fig_cam_4models_grid.png
    └── quantitative/fig_cam_quantitative.png
```

Tất cả hình ảnh nằm trong:
`/home/tower2080/Documents/CVRP_xAI/results/figures/`

## Bước 3 — Cài đặt Overleaf

- Compiler: **XeLaTeX** (bắt buộc vì dùng fontspec + polyglossia)
- Biber: bật (cho bibliography)
- Main file: `main.tex`

## Bước 4 — Compile

Overleaf sẽ compile tự động. Nếu gặp lỗi:
1. Kiểm tra tất cả figure paths đúng
2. Đảm bảo compiler là XeLaTeX
3. Chạy lại 2 lần để cross-references đúng

## Lưu ý về hình ảnh

Một số hình ảnh khá lớn (6-7MB). Nếu Overleaf báo lỗi memory:
- Giảm \`savefig.dpi\` khi tạo figures (từ 180 xuống 100)
- Hoặc dùng \`[width=0.8\\textwidth]\` thay vì \`[width=\\textwidth]\`
