# Quy trình làm việc với Git

## Nhánh

| Nhánh | Mục đích |
|---|---|
| `main` | Luôn chạy được. Chỉ cập nhật qua pull request |
| `feat/<mô-tả>` | Tính năng mới, ví dụ `feat/attention-heatmap` |
| `fix/<mô-tả>` | Sửa lỗi, ví dụ `fix/baomoi-thumbnail` |
| `docs/<mô-tả>` | Chỉ sửa tài liệu |
| `exp/<mô-tả>` | Thử nghiệm mô hình, ví dụ `exp/warmup` |

Tên nhánh viết thường, không dấu, nối các từ bằng `-`.

## Commit

Theo [Conventional Commits](https://www.conventionalcommits.org/): `<loại>(<phạm vi>): <mô tả ngắn>`

| Loại | Dùng khi |
|---|---|
| `feat` | Thêm chức năng |
| `fix` | Sửa lỗi |
| `refactor` | Đổi cấu trúc code, không đổi hành vi |
| `docs` | Tài liệu |
| `test` | Thêm hoặc sửa test |
| `exp` | Kết quả thực nghiệm (`results/`) |
| `chore` | Cấu hình, `.gitignore`, `requirements.txt` |

Ví dụ:

```
feat(models): thêm mô hình concat làm baseline hợp nhất muộn
fix(signals): sửa mẫu "không thể bỏ qua"
exp: kết quả 4 mode × 3 seed với warmup 10%
```

Mỗi commit chỉ làm một việc. Không commit file `.pth`, `data/images/`, `venv/`, `.env` (đã có trong `.gitignore`).

## Quy trình một thay đổi

```powershell
git switch main
git pull
git switch -c feat/ten-tinh-nang
# ... sửa code ...
python -m pytest -m "not slow"
git add <file>
git commit -m "feat(app): ..."
git push -u origin feat/ten-tinh-nang
```

Sau đó mở pull request vào `main` trên GitHub, mô tả **làm gì, vì sao, đã kiểm tra thế nào**, và nhờ một thành viên khác review trước khi merge.

## Chia sẻ trọng số mô hình

File `.pth` rất nặng (330–860 MB mỗi file) nên không đưa lên GitHub. Hãy chia sẻ qua Google Drive và đặt vào `checkpoints/` với đúng tên `{mode}_seed{seed}.pth`.
