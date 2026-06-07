# OSN-PDEA

## 环境配置：

```bash
conda create -n osn-pdea python=3.8 -y
conda activate osn-pdea
pip install pyautogui pillow matplotlib pandas numpy
```

## 运行脚本：

```bash
python main.py
```

每一步程序都会输出操作指引。

## 结果评估

```bash
python scripts/statistical_tests.py
```

默认读取 `results/statistics/all_steps.csv` 和 `results/statistics_by_type/all_steps_*.csv`，输出：

- `results/statistical_tests/exposure_summary.csv`
- `results/statistical_tests/chi_square_tests.csv`
- `results/statistical_tests/permutation_tests.csv`
- `results/statistical_tests/dimension_mean_tests.csv`
- `results/statistical_tests/figures/`

常用参数：

```bash
python scripts/statistical_tests.py --permutations 10000 --count-mode binary
python scripts/statistical_tests.py --count-mode score
```
