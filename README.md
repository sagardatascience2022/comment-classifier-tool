
# Comment Category Predictor

This project is a web application built with Streamlit that uses a fine-tuned DistilBERT model to classify comments into different categories such as Praise/Support, Hate/Abuse, Threat, Emotional, Irrelevant/Spam, Constructive Criticism, and Question/Suggestion. The application allows users to predict the category of a single comment or upload a CSV/JSON file containing multiple comments for batch processing. For batch uploads, it also visualizes the distribution of predicted categories and provides suggested reply templates based on the classification.

## Tech Stack

*   **Python:** The primary programming language used.
*   **Streamlit:** For creating the interactive web application interface.
*   **Transformers (Hugging Face):** For loading and using the pre-trained and fine-tuned DistilBERT model and tokenizer.
*   **PyTorch:** The deep learning framework used by the Transformers library.
*   **Pandas:** For data manipulation, especially for handling CSV and JSON file uploads.
*   **Matplotlib & Seaborn:** For visualizing the category distribution of batch processed comments.

## How to Run the Application

### Prerequisites

*   Python 3.6 or higher
*   pip (Python package installer)

### Setup

1.  **Clone the repository (if applicable) or ensure you have the `app.py` and `synthetic_comments_dataset.csv` files.**
2.  **Install the required libraries:**
    ```bash
    pip install streamlit transformers torch pandas matplotlib seaborn datasets
    ```
    *(Note: `torch` might require specific installation instructions depending on your system and CUDA availability. Refer to the official PyTorch documentation for details.)*
3.  **Ensure you have the fine-tuned model weights saved in a `./results` directory.** If you have trained the model in a previous step in this environment, the `results` directory should exist with checkpoint folders (e.g., `./results/checkpoint-XYZ`). The `app.py` script attempts to load the latest checkpoint automatically. If no checkpoint is found, it will use the base pre-trained model, which will have lower accuracy on the specific comment categories.

### Running the App

1.  **Open your terminal or command prompt.**
2.  **Navigate to the directory where `app.py` is located.**
3.  **Run the Streamlit application using the following command:**
    ```bash
    streamlit run app.py
    ```
4.  **The application will open in your web browser.** If it doesn't open automatically, click on the local URL provided in the terminal output (usually `http://localhost:8501`).

## Examples

### Single Comment Prediction

1.  Open the application in your browser.
2.  Scroll to the "Predict a Single Comment" section.
3.  Enter a comment in the text area, e.g., "This is an amazing video!".
4.  Click the "Predict Single Comment" button.
5.  The predicted category will be displayed below the button.

### Batch File Processing (CSV/JSON)

1.  Open the application in your browser.
2.  Scroll to the "Predict Categories from a File (CSV or JSON)" section.
3.  Click the "Browse files" button.
4.  Select a CSV or JSON file that contains a column named `comment`. You can use the generated `synthetic_comments_dataset.csv` for testing.
5.  Once the file is uploaded, the application will process the comments, display a table with the original comments and their predicted categories, and show a bar chart visualizing the distribution of the predicted categories.
6.  Suggested reply templates for each comment in the batch will also be displayed.

