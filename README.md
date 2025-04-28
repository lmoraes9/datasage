# DataSage 🧙: Interactive Data Analysis & Machine Learning Platform

Welcome to DataSage! This web application provides a user-friendly, interactive environment 💻 for performing common data analysis and machine learning tasks directly in your browser. Built with **Python**, **Django**, **Pandas**, **Scikit-learn**, **Seaborn/Matplotlib**, and **Bootstrap**, DataSage aims to streamline the workflow from data loading to model evaluation 🚀.

Whether you're quickly exploring a new dataset 🔍, performing essential cleaning operations 🧹, or training and comparing predictive models 🧠, DataSage offers a suite of tools to accelerate the process. This project serves as both a practical utility and a portfolio piece demonstrating skills in full-stack web development, data manipulation, visualization, and applied machine learning.

## Key Features & Screenshots ✨

Here's a walkthrough of the main functionalities, illustrated by the project screenshots:

1.  **Landing Page & Data Input:**
    *   **Home Page:** The initial entry point featuring the DataSage title/logo 🧙 and a brief description. Users can either upload their own CSV/XLSX file using the `Select File` button or the drag-and-drop zone 📤, or load a sample dataset (Iris) to get started quickly.
      
        ![Landing Page Screenshot](https://github.com/user-attachments/assets/2f3ec7d0-b078-45c5-9af9-4f24c061230d)

2.  **Analysis Dashboard - Overview:**
    *   **Analysis Dashboard - Overview Tab (Top):** After loading data, the main dashboard appears. The "Overview" tab initially shows high-level dataset statistics (Rows, Columns, Missing %, Duplicates) and a preview of the first few rows 👀.
      
        ![Dashboard Overview Tab - Top Section](https://github.com/user-attachments/assets/a47f4012-bd74-4a90-a84e-acb1a59fbffb)

    *   **Analysis Dashboard - Overview Tab (Column Summaries):** Further down, the "Column Summaries" accordion provides detailed insights into each variable (type, missing %, statistics) and allows generating univariate visualizations (Histogram 📈 and Box Plot shown).
    
        ![Dashboard Overview Tab - Column Summaries & Plots](https://github.com/user-attachments/assets/74de9b2b-b142-4281-b6e6-7f99dfc3d9f4)

    *   **Analysis Dashboard - Overview Tab (Correlation):** Explore relationships between numeric variables by generating a Correlation Matrix heatmap 🔥.
    
        ![Dashboard Overview Tab - Correlation Matrix](https://github.com/user-attachments/assets/1ae10ee2-db41-4d97-b40d-602a612bf854)

    *   **Analysis Dashboard - Overview Tab (Scatter Plot):** Visualize the relationship between two specific numeric variables with a Scatter Plot ✨.
    
        ![Dashboard Overview Tab - Scatter Plot](https://github.com/user-attachments/assets/68178f2e-03c4-4368-915f-3859a882040e)

3.  **Data Cleaning:**
    *   **Analysis Dashboard - Data Cleaning Tab:** Tools for data preprocessing 🧹. Handle missing values, remove duplicates, and detect/cap outliers using the IQR method.
      
        ![Dashboard Data Cleaning Tab](https://github.com/user-attachments/assets/ce1a3626-d9a7-4cff-827f-1e51210aff69)

4.  **Analysis & Machine Learning:**
    *   **Analysis Dashboard - ML Tab:** Train and evaluate models ⚙️. Select the target variable, choose one or multiple ML models (Regression/Classification), configure the test split, and train! Results are displayed in a comparison table with key metrics and download links 📥 for the trained model pipelines (`.joblib`).
      
        ![Dashboard Machine Learning Tab](https://github.com/user-attachments/assets/5b85adab-cb9c-4ccb-9a5b-0fd6bb540a9d)

DataSage aims to provide a practical and visually informative platform for common data science tasks. Feel free to explore, upload your data, and experiment with the models!🧪



















