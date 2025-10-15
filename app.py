import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title='Breast Cancer Classifier', layout='centered')

@st.cache_resource
def load_model(path='best_model_breast_cancer.pkl'):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def main():
    st.title('Breast Cancer Classification')
    st.write('Simple demo app that loads a pre-trained model and predicts whether a sample is benign (1) or malignant (0).')

    model = load_model()
    if model is None:
        st.error('Model file `best_model_breast_cancer.pkl` not found in the app folder.\nPlease place the pickle file next to this `app.py` and reload.')
        st.stop()

    # Try to infer feature names and expected number of features from the loaded model
    feature_names = None
    n_features_expected = None
    try:
        # Check pipeline steps first
        if hasattr(model, 'named_steps'):
            for step in model.named_steps.values():
                if hasattr(step, 'feature_names_in_'):
                    feature_names = list(step.feature_names_in_)
                    break
                if hasattr(step, 'n_features_in_') and n_features_expected is None:
                    n_features_expected = int(step.n_features_in_)

        # Check top-level attributes if not found in steps
        if feature_names is None and hasattr(model, 'feature_names_in_'):
            feature_names = list(model.feature_names_in_)
        if n_features_expected is None and hasattr(model, 'n_features_in_'):
            n_features_expected = int(model.n_features_in_)

        # Fallbacks
        if feature_names is None and n_features_expected is not None:
            feature_names = [f'feature_{i+1}' for i in range(n_features_expected)]
        if n_features_expected is None and feature_names is not None:
            n_features_expected = len(feature_names)
        if feature_names is None and n_features_expected is None:
            # final fallback: assume 30 features (the breast_cancer dataset)
            n_features_expected = 30
            feature_names = [f'feature_{i+1}' for i in range(n_features_expected)]
    except Exception:
        # safe fallback
        n_features_expected = n_features_expected or 30
        feature_names = feature_names or [f'feature_{i+1}' for i in range(n_features_expected)]

    st.sidebar.header('Input options')
    input_mode = st.sidebar.radio('Select input mode', ['Manual single sample', 'Upload CSV (multiple rows)', 'Show example rows'])

    # Define default example row based on common breast cancer dataset features
    example_df = None
    # create a minimal example using mean features from the notebook logic if available
    default_columns = [
        'mean radius','mean texture','mean perimeter','mean area','mean smoothness',
        'mean compactness','mean concavity','mean concave points','mean symmetry','mean fractal dimension',
    ]

    if input_mode == 'Manual single sample':
        st.subheader('Enter feature values')
        st.write(f'Model expects {n_features_expected} features. Provide values using the inputs below or upload a CSV with matching column names.')

        # Use a form so inputs are submitted together
        with st.form('manual_form'):
            values = []
            with st.expander('Feature inputs (expand to edit)', expanded=True):
                for col in feature_names:
                    # use a stable key for Streamlit widgets
                    val = st.number_input(col, value=0.0, format='%.5f', key=f'man_{col}')
                    values.append(val)
            submitted = st.form_submit_button('Predict')

        if submitted:
            input_df = pd.DataFrame([values], columns=feature_names)
            try:
                preds = model.predict(input_df)
                proba = model.predict_proba(input_df)[:, 1]
            except Exception as e:
                # try numpy fallback
                try:
                    preds = model.predict(input_df.values)
                    proba = model.predict_proba(input_df.values)[:, 1]
                except Exception as e2:
                    st.error(f'Prediction failed: {e} / {e2}')
                    input_df = None
                    preds = None
                    proba = None

            if preds is not None:
                st.write('Prediction (0=malignant, 1=benign):', int(preds[0]))
                st.write(f'Predicted probability of benign (class 1): {proba[0]:.3f}')

    elif input_mode == 'Upload CSV (multiple rows)':
        st.subheader('Upload CSV with feature columns')
        uploaded = st.file_uploader('Upload CSV', type=['csv'])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.write('Preview of uploaded data:')
            st.dataframe(df.head())
            if st.button('Predict uploaded rows'):
                try:
                    preds = model.predict(df)
                    proba = model.predict_proba(df)[:, 1]
                except Exception as e:
                    st.error(f'Error when predicting uploaded data: {e}')
                else:
                    out = df.copy()
                    out['predicted'] = preds
                    out['prob_benign'] = proba
                    st.write('Predictions:')
                    st.dataframe(out.head(50))
                    csv = out.to_csv(index=False).encode('utf-8')
                    st.download_button('Download predictions CSV', data=csv, file_name='predictions.csv')

    else:  # Show example rows
        st.subheader('Example dataset preview')
        # try to construct example from the common breast cancer CSV if present
        csv_path = 'breast_cancer_df.csv'
        if os.path.exists(csv_path):
            ex = pd.read_csv(csv_path)
            st.write('Showing first 5 rows of `breast_cancer_df.csv` (created by the notebook)')
            st.dataframe(ex.head())
            if st.button('Predict first 5 rows'):
                X = ex.drop(columns=['target'], errors='ignore')
                preds = model.predict(X)
                proba = model.predict_proba(X)[:, 1]
                out = X.copy()
                out['predicted'] = preds
                out['prob_benign'] = proba
                st.dataframe(out.head())
        else:
            st.info('No example CSV `breast_cancer_df.csv` found in the folder. You can generate it from the notebook or upload your own CSV.')

    st.markdown('---')
    st.write('Model info:')
    try:
        st.write(f'Type: {type(model)}')
        if hasattr(model, 'named_steps'):
            st.write('Pipeline steps: ' + ', '.join(model.named_steps.keys()))
    except Exception:
        pass


if __name__ == '__main__':
    main()
