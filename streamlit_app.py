import streamlit as st
import pandas as pd
from io import BytesIO

st.set_page_config(page_title='ML Test Dataset Generator', layout='wide',page_icon=':material/data_object:')

def match_distribution(training_data, testing_data):
    # Compute the desired distribution from the training data
    class_counts = training_data['gold_label'].value_counts()
    total_training_samples = len(training_data)
    desired_distribution = (class_counts / total_training_samples).to_dict()

    # Calculate sample counts for each class
    total_scraped_samples = len(testing_data)
    sample_counts = {label: int(total_scraped_samples * proportion) for label, proportion in desired_distribution.items()}

    # Create a new dataset matching the desired distribution
    matched_distribution_data = pd.DataFrame()
    for label, count in sample_counts.items():
        class_data = testing_data[testing_data['gold_label'] == label]
        matched_class_data = class_data.sample(n=min(count, len(class_data)), random_state=42)
        matched_distribution_data = pd.concat([matched_distribution_data, matched_class_data])

    # Adjust the sample size to exactly match the desired distribution
    matched_distribution_data = matched_distribution_data.sample(frac=1, random_state=42).reset_index(drop=True)

    return matched_distribution_data

# Streamlit UI
st.title("Match Data Distribution")

st.markdown("Upload the student's training dataset to generate a testing set with a matching distribution.")

# File uploaders
training_file = st.file_uploader("Upload Training Data CSV", type=["csv"])

if training_file:
    # Load the data
    training_data = pd.read_csv(training_file)
    testing_data = pd.read_csv('testing_data.csv')

    # Validate 'gold_label' column existence
    if 'gold_label' not in training_data.columns or 'gold_label' not in testing_data.columns:
        st.error("The files must contain a 'gold_label' column.")
    else:
        col1,col2 = st.columns(spec=2)
        # Process and display distributions
        with col1:
            st.subheader("Training Data Distribution")
            class_counts = training_data['gold_label'].value_counts(normalize=True) * 100
            class_counts_sorted = class_counts.sort_index().apply(lambda x: f'{x:.2f}%')
            st.dataframe(class_counts_sorted)

        matched_data = match_distribution(training_data, testing_data)
        with col2:
            st.subheader("Matched Data Distribution")
            matched_distribution = matched_data['gold_label'].value_counts(normalize=True) * 100
            matched_distribution_formatted = matched_distribution.apply(lambda x: f'{x:.2f}%')
            st.dataframe(matched_distribution_formatted)

        # Download link for matched data
        buffer = BytesIO()
        matched_data.to_csv(buffer, index=False)
        buffer.seek(0)
        st.download_button(
            label="Download Matched CSV",
            data=buffer,
            file_name="matched_data.csv",
            mime="text/csv"
        )
