| Package                | Version          |
|------------------------|------------------|
| `OS`                   | `ubuntu 24.04.1` |
| `cuda`                 | `11.8`           |
| `cudnn`                | `9`              |
| `python`               | `3.12.3`         |
| `numpy`                | `2.1.2`          |
| `MedPy`                | `0.5.2`          |
| `torch`                | `2.5.0`          |
| `torchaudio`           | `2.5.0`          |
| `torchmetrics`         | `1.7.0`          |
| `torchvision`          | `0.20.0`         |
| `pytorch-lightning`    | `2.5.1`          |

U-Mamba is an optional CUDA-only baseline. It was tested with the base
`torch==2.5.0+cu118` environment above and these extension versions:

| Optional package       | Tested version   |
|------------------------|------------------|
| `causal-conv1d`        | `1.4.0`          |
| `mamba-ssm`            | `2.2.4`          |

```
cd ~
python3 -m venv VENV
source VENV/bin/activate
(VENV): cd ~/UKAD/requirements
(VENV): pip install -r ./requirements.txt
```

For a CUDA 11.8 U-Mamba environment, install the optional extensions only
after PyTorch is present so their build can import it:

```bash
python -m pip install --no-build-isolation -r ./umamba-cu118.txt
```

CPU-only users should skip this optional file; it is not required by any other
architecture.
