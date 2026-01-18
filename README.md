# Neural Simulation of Robot Soccer Kit Dynamics

**Description:** This project explores the replacement of an analytical dynamics simulator of the Robot Soccer Kit(https://robot-soccer-kit.github.io/) by a learning-based neural simulator trained from real-world data. The goal is to reduce the sim-to-real gap.


<img src="https://via.placeholder.com/900x300.png?text=Video+illustrating+your+project,+or+picture"> 

[📖 User documentation](docs/user) • [👨‍💻 Developer documentation](docs/developer) • [📈 Project report](docs/report) • [📚 Bibliography](docs/bibliography) • [⚠️ Risk Analysis](docs/risk)
  
## 📄 This project in short
This paragraph is for the visitors who fly over your work and cannot read the whole documentation. They dislike long texts.

Be **concise** and **convincing** to show the potential of your project. Be **honest** and list the limitations.  

* The context and the intented users

This project is developed in an academic context around the Robot Soccer Kit (RSK) (https://robot-soccer-kit.github.io/), an open robotics platform created by the Rhoban laboratory. It targets students and researchers interested in robotics simulation and sim-to-real approaches.

* The problems solved by your project

The project introduces a learning-based dynamics simulator trained from real robot data, providing a data-driven alternative to purely analytical simulation within the RSK ecosystem.

* How it solves them

A neural dynamics model is trained from trajectories collected on physical RSK robots using an external vision system, and integrated into the existing simulator pipeline. The approach focuses on learning motion dynamics while preserving compatibility with the original RSK software stack.

Current limitations include unmodeled robot–robot collisions, non-learned ball dynamics.

## 🚀 Quickstart (if relevant)

* **Install instructions**: List of software/hardware dependencies, and instructions to install them if relevant
* **Launch instructions**: Few lines of code to launch the main feature of your project

The following steps allow you to run the Robot Soccer Kit simulator with the neural dynamics model enabled.

```bash
# From the root of the repository
python -m venv venv

# Linux / macOS
source venv/bin/activate
# Windows
.\venv\Scripts\Activate.ps1

pip install -e src/robot-soccer-kit[gc]
pip install -e src/rsk-neural-simulator

game_controller --simulated

If this is written in user or dev docs, provide links.

## 🔍 About this project

|       |        |
|:----------------------------:|:-----------------------------------------------------------------------:|
| 💼 **Client**                |  Name of your Client *(1)*                                              |
| 🔒 **Confidentiality**       | **Public** or **Private** *(1)*                                         |
| ⚖️ **License**               |  [Choose a license](https://choosealicense.com/) *(1)*                  |
| 👨‍👨‍👦 **Authors**               |  César LARRAGUETA, Olivier ROUAULT, Antony Thiery Student names, with a link to their social media profile or website    |


*(1) Refer to your client to make a choice. Then update the repository accordingly: the visibility in the settings and replace the [LICENSE](./LICENSE) file.*

## ✔️ Additional advices

* Do not make **passwords** and secret keys public. If you have to, replace it by a random string and a warning in the doc telling to replace it
* Avoid **long sentences**. Often, bullet points are easier to read
* **Illustrate** your reports. Use colored plots, schematics and pictures. But do not abuse of them
* Do not **duplicate** information. If it may be relevant at several places, make links
* **English** is the universal langage worldwide. Write all engineering documents in English
* Choose carefully **what sections** apply to your project and delete/add anything from the template that you think relevant
* Remove anything that would **pollute** reading, including these instructions and irrelevant sections
