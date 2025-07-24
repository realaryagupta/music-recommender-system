# 🎵 SoundBit: Personalized Music Recommender System

<div align="center">

![SoundBit Logo](https://img.shields.io/badge/SoundBit-Music%20Recommender-blueviolet?style=for-the-badge&logo=music)

[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

*A sophisticated, end-to-end music recommender system with hybrid AI models*

</div>

---

## 📑 Table of Contents

- [🎯 Introduction](#-introduction)
- [✨ Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [🛠️ Technologies Used](#️-technologies-used)
- [⚙️ Setup and Installation](#️-setup-and-installation)
- [🚀 Running the Project Locally](#-running-the-project-locally)
- [📖 Usage](#-usage)
- [📸 Screenshots](#-screenshots)
- [🔮 Future Enhancements](#-future-enhancements)
- [📞 Contact](#-contact)
- [📄 License](#-license)

---

## 🎯 Introduction

**SoundBit** is a sophisticated, end-to-end music recommender system designed to provide highly personalized song suggestions to users. Leveraging a large-scale dataset, the system integrates three distinct yet complementary recommendation methodologies: **content-based filtering**, **collaborative filtering**, and a robust **hybrid model**. This multi-faceted approach ensures superior recommendation accuracy and enhances user satisfaction by catering to diverse musical tastes and discovery patterns.

This project demonstrates expertise in the full machine learning lifecycle, from large-scale data engineering and model development to full-stack web application . It showcases the ability to build scalable, performant, and user-friendly AI-powered solutions.

---

## ✨ Features

### 🤖 **Hybrid Recommendation Model**
Combines the strengths of content-based and collaborative filtering for optimal suggestion quality.

### 🎯 **Content-Based Filtering**
Recommends songs similar to a user's input based on song attributes (e.g., genre, tempo, mood).

### 👥 **Collaborative Filtering**
Identifies songs preferred by users with similar listening histories.

### 🔍 **Fuzzy Matching for Input**
Robustly matches user-provided song titles and artist names, offering suggestions for typos or variations.

### ⚡ **Scalable Data Processing**
Utilizes Dask for efficient manipulation and processing of large datasets, ensuring performance even with extensive music libraries.

### 🖥️ **Interactive Web Interface**
A dynamic and intuitive frontend built with HTML, CSS, and JavaScript for seamless user interaction.

### 🚀 **FastAPI Backend**
A high-performance Python API to serve recommendations, ensuring quick response times.

---

## 🏗️ Architecture

The SoundBit system follows a **client-server architecture**:

### Frontend
A static web application built with HTML, CSS, and JavaScript, providing the user interface for inputting song details and displaying recommendations.

### Backend
A FastAPI application that hosts the recommendation models. It receives user requests, processes them using the integrated recommendation algorithms, and returns personalised song suggestions.

### Data & Models
Large datasets and pre-trained models (TF-IDF transformer, TF-IDF matrix, interaction matrix, track IDs, cleaned song metadata) are stored and loaded by the FastAPI backend. Git Large File Storage (LFS) is used to manage large model and data files within the repository.

```
+-------------------+       HTTP/JSON       +-------------------+       Model/Data Loading       +---------------------+
|                   | <-------------------> |                   | <----------------------------> |                     |
|  User (Browser)   |                       |   Local Frontend  |                                |  Local Backend      |
| (index.html,      |                       | (HTML, CSS, JS)   |                                | (FastAPI, Python,   |
|  script.js,        |                       |                   |                                |  Dask, Scipy, Pandas)|
|  style.css)       |                       |                   |                                |                     |
+-------------------+                       +-------------------+                                +---------------------+
                                                                                                        |
                                                                                                        | Large Files (LFS)
                                                                                                        V
                                                                                                  +---------------------+
                                                                                                  |  GitHub Repository  |
                                                                                                  |  (data/, models/)   |
                                                                                                  +---------------------+
```

---

## 🛠️ Technologies Used

### Backend
| Technology | Purpose |
|------------|---------|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | Core programming language (3.9+) |
| ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white) | Web framework for building the API |
| ![Uvicorn](https://img.shields.io/badge/Uvicorn-4B8BBE?style=flat) | ASGI server for running the FastAPI application |
| ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) | Data manipulation and analysis |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) | Numerical computing, especially for matrix operations |
| ![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white) | Scientific computing, including sparse matrix operations |
| ![Joblib](https://img.shields.io/badge/Joblib-F37626?style=flat) | For saving and loading Python objects |
| ![Dask](https://img.shields.io/badge/Dask-FDA061?style=flat&logo=dask&logoColor=white) | Large data processing with Pandas/NumPy |
| **Difflib** | For fuzzy string matching of song and artist names |

### Frontend
| Technology | Purpose |
|------------|---------|
| ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white) | Structure of the web page |
| ![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=css3&logoColor=white) | Styling and animations |
| ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black) | Dynamic behavior and API communication (ES6+) |
| ![Font Awesome](https://img.shields.io/badge/Font%20Awesome-339AF0?style=flat&logo=fontawesome&logoColor=white) | Icons |

### Repository Management
- **Git Large File Storage (LFS)**: For managing large model and data files in the repository

---

## ⚙️ Setup and Installation

Follow these steps to get the project running on your local machine.

### Prerequisites

- ![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat&logo=python) or higher
- ![Git](https://img.shields.io/badge/Git-F05032?style=flat&logo=git&logoColor=white)
- ![pip](https://img.shields.io/badge/pip-3776AB?style=flat&logo=pypi&logoColor=white) (Python package installer)

### 📥 Clone the Repository

First, clone the project repository to your local machine. Ensure you have Git LFS installed and configured before cloning, as the project contains large model and data files.

```bash
# Install Git LFS if you haven't already
git lfs install

# Clone the repository
git clone https://github.com/[Your_GitHub_Username]/musiccccc.git # Replace with your actual repo URL
cd musiccccc
```

### 🐍 Set up Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows
```

### 📦 Install Dependencies

With your virtual environment activated, install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project Locally

Once dependencies are installed, you can run the backend and frontend locally.

### 🔧 Start the Backend

Navigate to the root directory of your project (`musiccccc/`) and start the FastAPI backend using Uvicorn:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

- The `--reload` flag enables auto-reloading on code changes (useful for development)
- The backend will be accessible at `http://127.0.0.1:8000` or `http://localhost:8000`
- You should see output indicating that the server is running and models are loaded successfully

### 🌐 Access the Frontend

After the backend is running, open the frontend in your web browser:

1. Navigate to the `frontend` directory in your file explorer
2. Double-click `index.html` to open it in your default browser

> **📝 Note on CORS:** If you encounter CORS errors when running locally by opening `index.html` directly, ensure that your `backend/main.py` has the CORSMiddleware configured to allow null or `file://` origins, or consider using a simple local web server for the frontend (e.g., `python -m http.server 3000` from the frontend directory).

---

## 📖 Usage

1. **🌐 Open the frontend** in your browser
2. **✏️ Enter the Song Title** and **Artist Name** in the input fields
3. **🔢 Select the desired Number of Recommendations**
4. **🎯 Click the "Get Recommendations" button**

The system will display personalized song recommendations categorized by **Hybrid**, **Content-Based**, and **Collaborative** filtering. If an exact match isn't found, it will suggest similar songs.

---

## 📸 Screenshots

> *(Placeholder for Screenshots/Demo Video)*

### 🖼️ Main Input Interface

![](https://github.com/realaryagupta/music-recommender-system/blob/master/assets/Landing%20Page.png?raw=true)

### 🎵 Recommendations Display

**Love Story by Taylor Swift**

Collaborative Filtering Recommendation

![](https://github.com/realaryagupta/music-recommender-system/blob/master/assets/Love%20Story%20Collaborative.png?raw=true)

Content - Based Recommendation
![](https://github.com/realaryagupta/music-recommender-system/blob/master/assets/Love%20Story%20Content.png?raw=true)

Hybrid Recommendations
![](https://github.com/realaryagupta/music-recommender-system/blob/master/assets/Love%20Story%20Hybrid.png?raw=true)


### 💡 Suggestions for Mismatched Input

![](https://github.com/realaryagupta/music-recommender-system/blob/master/assets/Viva%20la%20Vida%20Content.png?raw=true)

![](https://github.com/realaryagupta/music-recommender-system/blob/master/assets/Viva%20la%20Vida.png?raw=true)

### 🎬 Demo Video
*Link to Demo Video (Optional): [Watch a quick demo here!](https://drive.google.com/file/d/1ONq_JyuZOAYIQdtHtdOFpaZP2EJYh5ax/view?usp=sharing)*

---

## 🔮 Future Enhancements

### 🔐 **User Authentication**
Implement user login to save listening history and provide more personalized recommendations over time.

### 👍 **Feedback Mechanism**
Allow users to rate recommendations to further refine the models.

### ⚡ **Real-time Recommendations**
Explore techniques for faster, more dynamic recommendations.

### 🌐 **More Data Sources**
Integrate with external music APIs (e.g., Spotify API) for richer metadata and direct playback.

### 🎨 **Advanced UI/UX**
Enhance the frontend with more interactive elements, filtering, and sorting options.

### 🔄 **Model Retraining Pipeline**
Automate the process of retraining models with new data.

---

## 📞 Contact

For any questions or collaborations, feel free to reach out:

**Arya Gupta**: [GitHub](https://github.com/realaryagupta)

---

## 📄 License

This project is licensed under the **MIT License**

---

<div align="center">

*Don't forget to ⭐ this repo if you found it helpful!*

</div>
