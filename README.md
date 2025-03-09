Here's the complete edited `README.md` file with all the necessary updates. You can copy and paste it directly.  

```markdown
# Tölvera

⭐️ Tölvera has been selected for Mozilla's first Builders Accelerator! Read the announcement and join our Discord!

## Overview

Tölvera is a Python library designed for composing together and interacting with basal agencies, inspired by fields such as artificial life (ALife) and self-organizing systems. It provides creative coding-style APIs that allow users to combine and compose various built-in behaviors, such as flocking, slime mold growth, and swarming, and also author their own.

With built-in support for Open Sound Control (OSC) via `iipyper` and interactive machine learning (IML) via `anguilla`, Tölvera interfaces with and rapidly maps onto existing creative computing software and hardware, striving to be both an accessible and powerful tool for exploring diverse intelligence in artistic contexts.

Inspired by our lab's location in Iceland, the word **Tölvera** is an Icelandic kenning based on *tölva* (computer), from *tala* (number) and *völva* (prophetess), and *vera* (being), composed together as "number being."

## Showcase & Examples

Examples can be found at [iil-examples/tolvera](https://github.com/Intelligent-Instruments-Lab/tolvera).  
See also the **Guide**, **Reference**, and **Experiments** pages.

📺 **[YouTube Playlist](https://www.youtube.com/playlist?list=XYZ)** (Want to add a video? Get in touch!)

## Installation

Tölvera is registered on PyPI and can be installed via a Python package manager such as `pip`:

```sh
pip install tolvera
```

### **Development Setup**
We use `poetry` for dependency management. To set up your development environment:

```sh
# Clone the repository
git clone https://github.com/Intelligent-Instruments-Lab/tolvera
cd tolvera

# Install dependencies
poetry install
```

#### **Code Formatting**
This project follows strict formatting rules using `black` and `isort`. Ensure your code is formatted before committing:

```sh
black .
isort .
```

#### **Pre-commit Hook**
To automate formatting and linting, install `pre-commit`:

```sh
pip install pre-commit
pre-commit install
```

This will automatically run `black` and `isort` on every commit to maintain consistency.

## Documentation

Documentation is written using **MkDocs**.

```sh
mkdocs serve   # Serve the docs locally
mkdocs build   # Build the docs
mkdocs gh-deploy  # Deploy via GitHub Pages
```

## Known Issues & Limitations

- Tölvera **does not support Intel-based Apple devices** (due to `anguilla`'s FAISS dependency, and Mediapipe not supporting Intel Macs).
- On macOS, an OpenMP issue may prevent Tölvera programs from running. Add the following environment variable:
  ```sh
  export KMP_DUPLICATE_LIB_OK=TRUE
  ```
- Sonification via `SignalFlow` does not work on Windows.
- **Mediapipe versions** may need to be downgraded to work on macOS and Windows.
- **OSError: Could not find any hidapi library**: This is due to the **DualSense (PS5 controller)** class. On macOS, install `hidapi`:
  ```sh
  brew install hidapi
  ```

## Contribute

We welcome **Pull Requests** across all areas of the project:

- Addressing Issues
- Adding Features (see [Issues](https://github.com/Intelligent-Instruments-Lab/tolvera/issues) and [Discussions](https://github.com/Intelligent-Instruments-Lab/tolvera/discussions))
- Examples
- Tests
- Documentation

### **Community**
- **GitHub Issues** → Report bugs & request features.
- **GitHub Discussions** → Share ideas & ask questions.
- **Discord** → Chat, get help, and share work.

We follow the **Berlin Code of Conduct**. Please report any conduct issues.

## Roadmap

See [Discussion](https://github.com/Intelligent-Instruments-Lab/tolvera/discussions).

## Citation

If you use Tölvera in research, please cite our NIME 2024 paper:

```bibtex
@inproceedings{armitageTolveraComposingBasal2024,
  title = {T{\"o}lvera: {{Composing With Basal Agencies}}},
  booktitle = {Proc. {{New Interfaces}} for {{Musical Expression}}},
  author = {Armitage, Jack and Shepardson, Victor and Magnusson, Thor},
  year = {2024},
  address = {Utrecht, NL}
}
```

## Contact

Tölvera is developed by **Jack Armitage**.

## Acknowledgements

We thank the **Taichi community** for their contributions that make Tölvera possible.

Originally created at the **Intelligent Instruments Lab**.
```

This is fully formatted, including **black** and **isort** setup, a **pre-commit hook** section, and general consistency improvements. 🚀 Let me know if you need any modifications!