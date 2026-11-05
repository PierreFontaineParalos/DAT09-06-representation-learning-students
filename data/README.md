# Datasets

Everything here is built by one command:

```bash
python data/prepare_data.py --all
```

The build is deterministic: same seed, same files, on any machine. Run it once
after installing the requirements, before session 2.

```
data/
├── prepare_data.py    the build
├── raw/               downloads — git-ignored, rebuildable
└── processed/         what the notebooks actually read
```

`python data/prepare_data.py --check` reports what is present.

---

## Fashion-MNIST — sessions 1, 2, 3

Zalando's drop-in replacement for MNIST: 70,000 greyscale 28×28 images of
clothing items across 10 classes.

| | |
|---|---|
| **Source** | `torchvision.datasets.FashionMNIST` (downloads itself, no account) |
| **Size** | ~30 MB |
| **Split** | 60,000 train / 10,000 test |
| **Licence** | MIT |
| **Citation** | Xiao, Rasul & Vollgraf (2017), *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms* |

**Why this dataset.** It trains in seconds on a CPU, it is visually
interpretable — you can *look* at a reconstruction and judge it — and these are
product photographs, which is what the course is about.

The ten classes: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt,
Sneaker, Bag, Ankle boot.

**How it is used.** Sessions 1–3 train autoencoders, VAEs and a contrastive
encoder on the *images only*. Labels are loaded, but used **exclusively for
evaluation** — probes, clustering metrics, colouring scatter plots. No model in
sessions 1–3 ever sees a label during training.

### Produced files

| File | Shape | Contents |
|---|---|---|
| `processed/fmnist_train.npz` | `(60000, 28, 28)` uint8 + `(60000,)` int64 | images, labels, classes |
| `processed/fmnist_test.npz` | `(10000, 28, 28)` uint8 + `(10000,)` int64 | images, labels, classes |

Load one with:

```python
import numpy as np
d = np.load("data/processed/fmnist_train.npz")
images, labels, classes = d["images"], d["labels"], d["classes"]
```

---

## Later sessions

Sessions 4–6 use real product photographs with titles and categories. Running a
vision model over thousands of them is not something a classroom of laptops can
do in a lab slot, so that data is prepared once and added to this repository.
There will be nothing more for you to build.
