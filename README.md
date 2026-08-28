# AI Challenge 2026 HCMC

Inference pipeline and training/evaluation setup.

## Setup
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure settings in `configs/default.yaml`
3. Place data in `data/`


##Chạy tools:
###tool 1:
. python src/tools/check_project_encoding.py --root .
Sau khi commit checkpoint mới chạy:
. python src/tools/check_project_encoding.py --root . --fix
. git diff

###tool 2:
. python src/tools/audit_dataset.py --config configs/default.yaml --output output/dataset_audit_full.json