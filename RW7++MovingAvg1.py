import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Default downsampling factor
DOWN_SAMPLE_DEFAULT = 175


COLORS = {
    f"Rotation {i}": f'rgba({(i*35)%256}, {(i*85)%256}, {(i*125)%256}, 0.5)' for i in range(1, 26)
}


def rgba_to_matplotlib_color(rgba_str):
    rgba_values = rgba_str.strip('rgba()').split(',')
    rgba_values = [float(val) / 255.0 if i < 3 else float(val) for i, val in enumerate(rgba_values)]
    return tuple(rgba_values)

# Function to calculate moving average
def calculate_moving_average(data, window_size):
    
    avg = np.convolve(data, np.ones(window_size) / window_size, mode='same')
    avg[:window_size // 2] = np.nan  
    avg[-window_size // 2:] = np.nan  
    return avg

def process_dataset(df, position_col, torque_col, downsample_factor):

    start_index = df[df[position_col].apply(lambda x: int(x) == 2)].index.min()
    if pd.isna(start_index):
        return None  

    df = df.iloc[start_index:].reset_index(drop=True)

    rotations = []
    last_position = df[position_col].iloc[0] 
    rotation_data = []

    for _, row in df.iterrows():
        position = row[position_col]
        torque = row[torque_col]

        if position - last_position >= 6.28318531:  
            rotations.append(rotation_data)
            rotation_data = []
            last_position = position
        rotation_data.append((position, torque))

    if rotation_data:  
        rotations.append(rotation_data)

    downsampled_rotations = {}
    for i, rotation in enumerate(rotations):
        positions = [p for p, _ in rotation]
        torques = [t for _, t in rotation]

        position_start = positions[0]
        relative_positions = [p - position_start for p in positions]

        degrees_positions = np.degrees([p % (2 * np.pi) for p in relative_positions])

        downsampled_torques = [
            np.mean(torques[j:j + downsample_factor]) for j in range(0, len(torques), downsample_factor)
        ]

        max_degrees = degrees_positions[-1]
        downsampled_positions = np.linspace(0, max_degrees, len(downsampled_torques))

        downsampled_rotations[f"Rotation {i+1}"] = {
            "positions": downsampled_positions,
            "torques": downsampled_torques
        }

    return downsampled_rotations


def plot_data_with_global_moving_avg(rotations, moving_avg_window):
    plt.figure(figsize=(10, 6))

    all_positions = []
    all_torques = []

    for label, data in rotations.items():
        color = rgba_to_matplotlib_color(COLORS.get(label, 'rgba(0, 0, 0, 0.5)'))
        positions = data["positions"]
        torques = data["torques"]

        positions = np.array(positions) - np.array(positions[0])  
        all_positions.extend(positions)
        all_torques.extend(torques)

        plt.plot(positions, torques, label=f"{label}", color=color, alpha=0.5)

    sorted_indices = np.argsort(all_positions)
    all_positions = np.array(all_positions)[sorted_indices]
    all_torques = np.array(all_torques)[sorted_indices]

    moving_avg = calculate_moving_average(all_torques, moving_avg_window)

    valid_indices = ~np.isnan(moving_avg)
    avg_positions = all_positions[valid_indices]
    moving_avg = moving_avg[valid_indices]

    plt.plot(avg_positions, moving_avg, color='red', linewidth=2.5, label="Global Moving Avg")

    plt.title(f"Torque vs. Position for {uploaded_file.name}")
    plt.xlabel("Position [°]")
    plt.ylabel("Torque")
    plt.xticks(np.arange(0, 361, 20))
    plt.ylim(-0.5, -0.2)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    st.pyplot(plt.gcf())

# Streamlit App UI
st.title("Torque and Position Analyzer")
st.write("Upload datasets")

downsample_factor = st.sidebar.slider("Downsampling Size", min_value=10, max_value=400, value=DOWN_SAMPLE_DEFAULT, step=5)

moving_avg_window = st.sidebar.slider("Moving Average Window Size", min_value=1, max_value=100, value=20, step=1)

uploaded_files = st.file_uploader("Upload CSV Files", type=["csv"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.subheader(f"File: {uploaded_file.name}")
        df = pd.read_csv(uploaded_file, delimiter='\t')

        df = df.drop(columns=['Unnamed: 3'], errors='ignore')

        position_col = "ActualPosition [rad]"
        torque_col = "Actual Torque [of nominal]"

        if position_col not in df.columns or torque_col not in df.columns:
            st.error(f"File {uploaded_file.name} does not contain the required columns: {position_col}, {torque_col}")
            continue

        timeframes = process_dataset(df, position_col, torque_col, downsample_factor)

        if timeframes is None:
            st.warning(f"File {uploaded_file.name} does not contain valid data starting with a minimum {position_col} value of 2.")
            continue

        plot_data_with_global_moving_avg(timeframes, moving_avg_window)
else:
    st.info("Upload one or more CSV files to begin.")

st.sidebar.write("Adjust the downsampling size and moving average window size using the sliders.")
