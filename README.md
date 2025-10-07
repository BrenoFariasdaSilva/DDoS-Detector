<div align="center">
  
# [DDoS-Detector.](https://github.com/BrenoFariasdaSilva/DDoS-Detector) <img src="[Icon-Image-URL](https://github.com/avastjs/cyber-icons/blob/main/logo.svg)"  width="3%" height="3%">

</div>

<div align="center">
  
---

Project-Description - @UPDATE.
  
---

</div>

<div align="center">

![GitHub Code Size in Bytes](https://img.shields.io/github/languages/code-size/BrenoFariasdaSilva/DDoS-Detector)
![GitHub Commits](https://img.shields.io/github/commit-activity/t/BrenoFariasDaSilva/DDoS-Detector/main)
![GitHub Last Commit](https://img.shields.io/github/last-commit/BrenoFariasdaSilva/DDoS-Detector)
![GitHub Forks](https://img.shields.io/github/forks/BrenoFariasDaSilva/DDoS-Detector)
![GitHub Language Count](https://img.shields.io/github/languages/count/BrenoFariasDaSilva/DDoS-Detector)
![GitHub License](https://img.shields.io/github/license/BrenoFariasdaSilva/DDoS-Detector)
![GitHub Stars](https://img.shields.io/github/stars/BrenoFariasdaSilva/DDoS-Detector)
![wakatime](https://wakatime.com/badge/github/BrenoFariasdaSilva/DDoS-Detector.svg)

</div>

<div align="center">
  
![RepoBeats Statistics](https://repobeats.axiom.co/api/embed/deca67a753c6ad283c2b87e95f2b676767739706.svg "Repobeats analytics image")

</div>

## Table of Contents
- [DDoS-Detector. ](#ddos-detector-)
  - [Table of Contents](#table-of-contents)
  - [Introduction - @UPDATE](#introduction---update)
  - [Requirements - @UPDATE](#requirements---update)
  - [Setup](#setup)
    - [1. Install Python](#1-install-python)
      - [Linux](#linux)
      - [macOS](#macos)
      - [Windows](#windows)
    - [2. Install `make` utility](#2-install-make-utility)
      - [Linux](#linux-1)
      - [macOS](#macos-1)
      - [Windows](#windows-1)
    - [3. Clone the repository](#3-clone-the-repository)
    - [4. Virtual environment (Strongly Recommended)](#4-virtual-environment-strongly-recommended)
    - [5. Install dependencies](#5-install-dependencies)
    - [6. Dataset - @UPDATE](#6-dataset---update)
  - [Results - @UPDATE](#results---update)
  - [How to Cite?](#how-to-cite)
  - [Contributing](#contributing)
  - [Collaborators](#collaborators)
  - [License](#license)
    - [Apache License 2.0](#apache-license-20)

## Introduction - @UPDATE

Detailed project description.

## Requirements - @UPDATE

Bullet points of the requirements, such as languages, libraries, tools, etc.

## Setup

Before running the project, ensure that both **Python** and the **make utility** are installed on your system. Follow the instructions below according to your operating system.

### 1. Install Python

The project requires **Python 3.9 or higher**.

#### Linux
On Debian/Ubuntu-based distributions:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip -y
```

On Fedora/RHEL-based distributions:

```bash
sudo dnf install python3 python3-venv python3-pip -y
```

Verify installation:

```bash
python3 --version
```

#### macOS
1. Install via Homebrew (recommended):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" # if Homebrew not installed
brew install python
```

2. Verify installation:

```bash
python3 --version
```

#### Windows
1. Download Python from the official website: [https://www.python.org/downloads/windows/](https://www.python.org/downloads/windows/)
2. Run the installer and check **“Add Python to PATH”**.
3. Verify installation:

```powershell
python --version
```

---

### 2. Install `make` utility

The `make` utility is used to automate tasks such as setting up the virtual environment and installing dependencies.

#### Linux
`make` is usually pre-installed. If not:

```bash
sudo apt install build-essential -y  # Debian/Ubuntu
sudo dnf install make -y            # Fedora/RHEL
make --version
```

#### macOS
`make` comes pre-installed with Xcode Command Line Tools:

```bash
xcode-select --install
make --version
```

#### Windows
1. Install via [Chocolatey](https://chocolatey.org/):

```powershell
choco install make
```

Or, install [GnuWin32 Make](http://gnuwin32.sourceforge.net/packages/make.htm).

2. Verify installation:

```powershell
make --version
```

---

### 3. Clone the repository

1. Clone the repository with the following command:

   ```bash
   git clone https://github.com/BrenoFariasDaSilva/DDoS-Detector.git
   cd DDoS-Detector
   ```

### 4. Virtual environment (Strongly Recommended)

With `make`:

```bash
make venv
source venv/bin/activate # Linux/macOS
venv\Scripts\activate # Windows
```

Or manually:

```bash
python -m venv venv
source venv/bin/activate # Linux/macOS
venv\Scripts\activate # Windows
```

### 5. Install dependencies

1. Install Python packages:

With `make`:

```bash
make dependencies
```

Or manually:

```bash
pip install -r requirements.txt
```

---

### 6. Dataset - @UPDATE

1. Download the dataset you want to use and place it in this project directory `(/DDoS-Detector)`, inside the `Datasets` folder.
A few of the used datasets can be found at:
   - [CICDDoS2019](https://www.unb.ca/cic/datasets/ddos-2019.html)
   - [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)
   - [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html)

## Results - @UPDATE

Discuss the results obtained in the project.

## How to Cite?

If you use the Repository-Name in your research, please cite it using the following BibTeX entry:

```
@misc{softwareDDoS-Detector:2025,
  title = {DDoS-Detector: A Tool for Detecting DDoS Attacks},
  author = {Breno Farias da Silva},
  year = {2025},
  howpublished = {https://github.com/BrenoFariasdaSilva/DDoS-Detector},
  note = {Accessed on October 6, 2026}
}
```

Additionally, a `main.bib` file is available in the root directory of this repository, in which contains the BibTeX entry for this project.

If you find this repository valuable, please don't forget to give it a ⭐ to show your support! Contributions are highly encouraged, whether by creating issues for feedback or submitting pull requests (PRs) to improve the project. For details on how to contribute, please refer to the [Contributing](#contributing) section below.

Thank you for your support and for recognizing the contribution of this tool to your work!

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**. If you have suggestions for improving the code, your insights will be highly welcome.
In order to contribute to this project, please follow the guidelines below or read the [CONTRIBUTING.md](CONTRIBUTING.md) file for more details on how to contribute to this project, as it contains information about the commit standards and the entire pull request process.
Please follow these guidelines to make your contributions smooth and effective:

1. **Set Up Your Environment**: Ensure you've followed the setup instructions in the [Setup](#setup) section to prepare your development environment.

2. **Make Your Changes**:
   - **Create a Branch**: `git checkout -b feature/YourFeatureName`
   - **Implement Your Changes**: Make sure to test your changes thoroughly.
   - **Commit Your Changes**: Use clear commit messages, for example:
     - For new features: `git commit -m "FEAT: Add some AmazingFeature"`
     - For bug fixes: `git commit -m "FIX: Resolve Issue #123"`
     - For documentation: `git commit -m "DOCS: Update README with new instructions"`
     - For refactorings: `git commit -m "REFACTOR: Enhance component for better aspect"`
     - For snapshots: `git commit -m "SNAPSHOT: Temporary commit to save the current state for later reference"`
   - See more about crafting commit messages in the [CONTRIBUTING.md](CONTRIBUTING.md) file.

3. **Submit Your Contribution**:
   - **Push Your Changes**: `git push origin feature/YourFeatureName`
   - **Open a Pull Request (PR)**: Navigate to the repository on GitHub and open a PR with a detailed description of your changes.

4. **Stay Engaged**: Respond to any feedback from the project maintainers and make necessary adjustments to your PR.

5. **Celebrate**: Once your PR is merged, celebrate your contribution to the project!

## Collaborators

We thank the following people who contributed to this project:

<table>
  <tr>
    <td align="center">
      <a href="#" title="defina o titulo do link">
        <img src="https://github.com/BrenoFariasdaSilva/DDoS-Detector/blob/main/.assets/Images/Github.svg" width="100px;" alt="My Profile Picture"/><br>
        <sub>
          <b>Breno Farias da Silva</b>
        </sub>
      </a>
    </td>
  </tr>
</table>

## License

### Apache License 2.0

This project is licensed under the [Apache License 2.0](LICENSE). This license permits use, modification, distribution, and sublicense of the code for both private and commercial purposes, provided that the original copyright notice and a disclaimer of warranty are included in all copies or substantial portions of the software. It also requires a clear attribution back to the original author(s) of the repository. For more details, see the [LICENSE](LICENSE) file in this repository.
