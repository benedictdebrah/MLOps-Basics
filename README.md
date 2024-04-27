# MLOps-Basics


This project is a basic introduction to the features of MLflow. It's designed to help me understand how MLflow works and how it can be used in machine learning projects.

## Project Overview

In this project, I've set up a simple machine learning workflow to explore the features of MLflow. <b>The goal wasn't to follow a rigorous data science workflow</b>, but rather to get things running and see how it goes.

## Features Explored

Here are some of the MLflow features I've explored in this project:

1. **Tracking**: I've used MLflow's tracking feature to log parameters, metrics, and artifacts. This helped me understand how MLflow can be used to keep track of different experiments in a systematic way.

2. **Projects**: I've set up this project as an MLflow project, which helped me understand how MLflow can be used to package machine learning code in a reusable and reproducible way.

3. **Models**: I've used MLflow's model feature to save and load models. This helped me understand how MLflow can be used to manage models and their versions.

## Running the Project

To run this project, you need to have Python and MLflow installed on your machine. Once you have these prerequisites, you can follow these steps:

1. Clone the repository to your local machine (install requirements).
2. Navigate to the project directory in the terminal.
3. Run the main script with the command: `python main.py`
4. After running the script, start the MLflow tracking server with the command: `mlflow ui`
5. After running `mlflow ui` the link that appears on the terminal click on it to access the platform



## Running the Project on DagsHub

To run this project on DagsHub, you need to have a DagsHub account. Once you have an account, you can follow these steps:

1. Fork the repository to your DagsHub account.
2. Set up the environment variables for MLflow tracking. You can do this by running the following commands in your terminal:

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/your_username/your_repo_name.mlflow
export MLFLOW_TRACKING_USERNAME=your_username
export MLFLOW_TRACKING_PASSWORD=your_password
```
3. Run the script `python main.py`


Please note that this project is a basic introduction to MLflow and does not follow a rigorous data science workflow.

## Future Work

I plan to explore more advanced features of MLflow and use it in more complex machine learning projects. I also plan to follow a more rigorous data science workflow in my future projects.


