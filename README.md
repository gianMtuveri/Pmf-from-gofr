# pmf-from-gofr

Hydration-consistent reconstruction of effective interaction potentials from radial distribution functions **g(r)**.

This package computes PMF-like profiles using an iterative closure scheme (Ornstein-Zernike equations) while preserving thermodynamic consistency at long range. It avoids artificial shifts (e.g. `g -= constant`) and instead enforces physically correct asymptotic behavior via tail normalization.

---

## ✨ Features

- Iterative closure-based PMF reconstruction  
- Tail normalization for long-range consistency  
- Savitzky–Golay smoothing  
- Command-line interface for batch processing  
- Python 3.10 compatible  

---

## 📦 Installation (Conda Recommended)

Create a clean environment:

```
conda create -n pmf310 python=3.10 pip -y
conda activate pmf310
```

Install the package:

```
pip install -e .
```

---

## 🚀 Usage

Example (RDF extending to 40 Å):

```
pmf-from-gofr \
    --names gofr_HW_BX gofr_HW_CX gofr_OW_Co \
    --in-dir ../ \
    --out-dir ./PMF_outputs \
    --tail-start 30 \
    --tail-method scale \
    --plots
```

If running without installation:

```
PYTHONPATH=src python -m pmf_from_gofr.cli \
    --names gofr_HW_BX \
    --in-dir ../ \
    --out-dir ./PMF_outputs \
    --tail-start 30
```

---

## 📂 Input Format

Each `.dat` file must contain at least:

```
x   g(r)   m(r)
```

Lines beginning with `@` or `#` are ignored.  
Only the first two columns are used.

---

## 📤 Output

For each input file:

```
PMF<name>.xvg
```

containing:

```
r   v(r)
```

Optional PNG plots are generated with `--plots`.

---

## 🔁 Reproducibility

Export environment:

```
conda env export --from-history > environment.yml
```

Recreate:

```
conda env create -f environment.yml
conda activate pmf310
```

---

## 📚 Scientific Context

The reconstruction philosophy is inspired by hydration-based free-energy frameworks and iterative closure methods from liquid-state theory and solvation thermodynamics.

---

## 📄 License

Research code for academic use.

---

## Citation

If used in academic work, please cite the associated doctoral thesis.
