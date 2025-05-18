import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_model():
    """Load the trained model and feature names."""
    # Get the absolute path to the models directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(os.path.dirname(current_dir), 'models')
    
    if not os.path.exists(models_dir):
        st.error(f"Models directory not found at: {models_dir}")
        return None, None
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.joblib')]
    if not model_files:
        st.error("No trained model found. Please train the model first.")
        return None, None
    
    model = joblib.load(os.path.join(models_dir, model_files[0]))
    feature_names = joblib.load(os.path.join(models_dir, 'feature_names.joblib'))
    
    # Print debugging information
    st.write("### Model Information")
    st.write(f"Model type: {type(model).__name__}")
    st.write("Features used by model:")
    st.write(feature_names)
    
    return model, feature_names

def main():
    st.set_page_config(
        page_title="Laptop Price Predictor",
        page_icon="💻",
        layout="wide"
    )
    
    st.title("💻 Laptop Price Predictor")
    st.write("""
    Enter the specifications of your laptop to get an estimated price.
    This model has been trained on various laptop configurations to provide accurate price predictions.
    """)
    
    # Load model and feature names
    model, feature_names = load_model()
    if model is None:
        return
    
    # Create three columns for input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Basic Specifications")
        brand = st.selectbox(
            "Brand",
            options=['Dell', 'HP', 'Lenovo', 'Asus', 'Acer', 'MSI', 'Apple', 'Other']
        )
        
        processor = st.selectbox(
            "Processor",
            options=[
                'Intel Core i3',
                'Intel Core i5',
                'Intel Core i7',
                'Intel Core i9',
                'AMD Ryzen 3',
                'AMD Ryzen 5',
                'AMD Ryzen 7',
                'AMD Ryzen 9',
                'Apple M1',
                'Apple M2',
                'Apple M2 Pro',
                'Apple M2 Max'
            ]
        )
        
        ram_gb = st.selectbox(
            "RAM (GB)",
            options=[4, 6, 8, 12, 16, 32, 64, 128],
            index=2  # Default to 8GB
        )
        
        ram_type = st.selectbox(
            "RAM Type",
            options=[
                'DDR4',
                'DDR5',
                'LPDDR4X',
                'LPDDR5',
                'Unified Memory'  # For Apple
            ]
        )
    
    with col2:
        st.subheader("Storage & Graphics")
        storage_gb = st.selectbox(
            "Storage (GB)",
            options=[
                128, 256, 512, 1024,  # 1TB
                2048,  # 2TB
                4096   # 4TB
            ],
            index=2  # Default to 512GB
        )
        
        storage_type = st.selectbox(
            "Storage Type",
            options=[
                'NVMe SSD',
                'SATA SSD',
                'HDD 5400RPM',
                'HDD 7200RPM'
            ]
        )

        gpu_brand = st.selectbox(
            "GPU Brand",
            options=[
                'NVIDIA',
                'AMD',
                'Intel',
                'Apple',
                'Other'
            ]
        )

        gpu_model = st.selectbox(
            "GPU Model",
            options=[
                # NVIDIA Options
                'NVIDIA RTX 4090',
                'NVIDIA RTX 4080',
                'NVIDIA RTX 4070',
                'NVIDIA RTX 4060',
                'NVIDIA RTX 3080',
                'NVIDIA RTX 3070',
                'NVIDIA RTX 3060',
                'NVIDIA GTX 1660 Ti',
                'NVIDIA MX550',
                'NVIDIA MX450',
                # AMD Options
                'AMD RX 7600M XT',
                'AMD RX 6800M',
                'AMD RX 6700M',
                'AMD RX 6600M',
                # Intel Options
                'Intel Arc A770M',
                'Intel Arc A730M',
                'Intel Arc A550M',
                'Intel Iris Xe',
                'Intel UHD',
                # Apple Options
                'Apple M1',
                'Apple M2',
                'Apple M2 Pro',
                'Apple M2 Max',
                # Integrated
                'Integrated Graphics'
            ]
        )

        gpu_memory = st.selectbox(
            "GPU Memory (GB)",
            options=[0, 2, 4, 6, 8, 12, 16, 24],
            index=2  # Default to 4GB
        )
    
    with col3:
        st.subheader("Display & System")
        display_size = st.selectbox(
            "Screen Size (inches)",
            options=[
                13.3,
                14.0,
                15.6,
                16.0,
                17.3
            ],
            index=2  # Default to 15.6"
        )
        
        resolution = st.selectbox(
            "Screen Resolution",
            options=[
                'HD (1366x768)',
                'FHD (1920x1080)',
                'QHD (2560x1440)',
                '4K UHD (3840x2160)',
                'Retina (2560x1600)',  # Common in MacBooks
                '2.8K (2880x1800)',
            ]
        )
        
        display_features = st.multiselect(
            "Display Features",
            options=[
                'IPS Panel',
                'Touch Screen',
                'Anti-Glare',
                'HDR',
                'High Refresh Rate (144Hz+)'
            ],
            default=[]
        )
        
        os = st.selectbox(
            "Operating System",
            options=[
                'Windows 11 Home',
                'Windows 11 Pro',
                'Windows 10 Home',
                'Windows 10 Pro',
                'macOS',
                'Linux Ubuntu',
                'Linux Fedora'
            ]
        )
        
        warranty = st.selectbox(
            "Warranty (Years)",
            options=[1, 2, 3, 4, 5],
            index=0  # Default to 1 year
        )
    
    # Create feature vector
    if st.button("Predict Price"):
        try:
            # Extract resolution values from the selected option
            resolution_str = resolution.split(' ')[1].strip('()')  # Extract "1920x1080" from "FHD (1920x1080)"
            width, height = map(int, resolution_str.split('x'))
            resolution_value = width * height
            
            # Map processor to basic categories for model
            processor_category = 'Intel' if 'Intel' in processor else 'AMD' if 'AMD' in processor else 'Apple'
            
            # Create feature dictionary with all required features
            features = {
                'spec_rating': 5.0 + (ram_gb/32) + (storage_gb/1024),  # Dynamic spec rating based on components
                'display_size': float(display_size),
                'warranty': float(warranty),
                'resolution': resolution_value,
                'storage_gb': float(storage_gb),
                'ram_gb': float(ram_gb),
                'gpu_memory_gb': float(gpu_memory),
                'brand_encoded': ['Dell', 'HP', 'Lenovo', 'Asus', 'Acer', 'MSI', 'Apple', 'Other'].index(brand),
                'processor_encoded': ['Intel', 'AMD', 'Apple'].index(processor_category),
                'OS_encoded': ['Windows', 'macOS', 'Linux'].index('Windows' if 'Windows' in os else 'macOS' if 'macOS' in os else 'Linux'),
                'Ram_type_encoded': ['DDR4', 'DDR5', 'LPDDR4X', 'LPDDR5', 'Unified Memory'].index(ram_type),
                'ROM_type_encoded': ['NVMe SSD', 'SATA SSD', 'HDD 5400RPM', 'HDD 7200RPM'].index(storage_type),
                'gpu_brand_encoded': ['NVIDIA', 'AMD', 'Intel', 'Apple', 'Other'].index(gpu_brand)
            }
            
            # Create DataFrame with all required columns in the correct order
            df = pd.DataFrame([features])
            
            # Debug information
            st.write("### Input Features")
            st.write(df)
            
            # Ensure columns are in the same order as during training
            try:
                X = df[feature_names]
                
                # Debug information
                st.write("### Processed Features")
                st.write(X)
                st.write("### Feature Names Required by Model:")
                st.write(feature_names)
                
                # Make prediction
                prediction = model.predict(X)[0]
                
                # Convert to USD (approximate rate: 1 USD = 83 INR)
                usd_price = prediction / 83
                
                # Display predictions
                st.success(f"Estimated Price: ₹{prediction:,.2f} (INR)")
                st.success(f"Estimated Price: ${usd_price:,.2f} (USD)")
                
                # Add a note about conversion rate
                st.info("Note: USD conversion uses approximate rate of ₹83 = 1 USD")
                
                # Display confidence interval (using model's feature importances if available)
                if hasattr(model, 'feature_importances_'):
                    st.write("### Feature Importance")
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    st.bar_chart(importance_df.set_index('Feature'))
            
            except KeyError as e:
                st.error(f"Missing required feature: {str(e)}")
                st.error("Features provided: " + ", ".join(df.columns))
                st.error("Features required: " + ", ".join(feature_names))
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Features required by model: " + ", ".join(feature_names))

if __name__ == "__main__":
    main() 