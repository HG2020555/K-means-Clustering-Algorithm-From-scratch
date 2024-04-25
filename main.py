import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy import stats


def preprocess_data(data):
    # Convert 'Gross' column to numeric, handle 'M' suffix
    data['Gross'] = data['Gross'].replace('[\$,M]', '', regex=True).astype(float) * 10 ** 6

    # Fill missing values in 'Metascore' column with median
    data['Metascore'] = data['Metascore'].fillna(data['Metascore'].median())

    # Fill missing values in 'Votes' column with 0
    data['Votes'] = data['Votes'].str.replace(',', '').astype(float)
    data['Votes'] = data['Votes'].fillna(0)

    # Fill missing values in 'Release Year' column with mode
    data['Release Year'] = data['Release Year'].fillna(data['Release Year'].mode()[0])

    # Drop rows with missing values in other columns
    data = data.dropna(subset=['IMDB Rating', 'Genre', 'Director', 'Cast'])

    return data


def detect_outliers_zscore(data):
    z_scores = np.abs(stats.zscore(data))
    threshold = 3
    outlier_indices = np.where(z_scores > threshold)[0]
    outliers = data.iloc[outlier_indices]
    return outliers


def initialize_centroids(data, k):
    np.random.seed(42)  # for reproducibility
    centroids = data.sample(n=k)
    return centroids.values


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def assign_clusters(data, centroids):
    clusters = []
    for i in range(data.shape[0]):
        distances = [euclidean_distance(data.iloc[i], centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return clusters


def update_centroids(data, clusters, k):
    new_centroids = []
    for cluster in range(k):
        cluster_points = data[np.array(clusters) == cluster]
        new_centroid = np.mean(cluster_points, axis=0)
        new_centroids.append(new_centroid)
        print()
    return np.array(new_centroids)


def kmeans(data, k, max_iters=100):
    #scaler = StandardScaler()
    #scaled_data = scaler.fit_transform(data)  # Standardize the data
    centroids = initialize_centroids(pd.DataFrame(data), k)
    print(centroids)

    for _ in range(max_iters):
        clusters = assign_clusters(pd.DataFrame(data), centroids)
        new_centroids = update_centroids(data, clusters, k)

        # Check for convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids
        print(centroids)


    return clusters, centroids


def preprocess_and_cluster(file_path, k, percentage):
    data = pd.read_csv(file_path)
    data = preprocess_data(data)
    data_to_cluster, outliers = partition_data(data[['IMDB Rating']], percentage)
    clusters, centroids = kmeans(data_to_cluster, k)
    return data_to_cluster, outliers, clusters, centroids


def partition_data(data, percentage):
    num_rows = len(data)
    num_rows_to_keep = int(num_rows * (1 - percentage))
    data_to_keep = data.sample(n=num_rows_to_keep)
    outliers = detect_outliers_zscore(data_to_keep[['IMDB Rating']])
    data_to_keep = data_to_keep.drop(index=outliers.index).reset_index(drop=True)
    return data_to_keep, outliers


def browse_file_path():
    file_path = filedialog.askopenfilename()
    file_path_entry.delete(0, tk.END)
    file_path_entry.insert(0, file_path)


def start_process():
    file_path = file_path_entry.get()
    k = int(k_entry.get())
    percentage = float(percentage_entry.get())
    data_to_cluster, outliers, clusters, centroids = preprocess_and_cluster(file_path, k, percentage)

    # Count number of records in each cluster
    cluster_counts = [len(data_to_cluster.iloc[np.where(np.array(clusters) == cluster)]) for cluster in range(k)]

    # Print the number of records after removing outliers
    result_text.config(state=tk.NORMAL)
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END,
                       "Number of records after removing outliers and clustering: {}\n\n".format(len(data_to_cluster)))

    # Display clusters
    sum=0
    for cluster in range(k):
        cluster_movies = data_to_cluster.iloc[np.where(np.array(clusters) == cluster)]
        result_text.insert(tk.END, "Number of records in Cluster {}: {}\n".format(cluster + 1, cluster_counts[cluster]))
        sum=sum + cluster_counts[cluster]
        result_text.insert(tk.END, "Cluster {} movies:\n".format(cluster + 1))
        result_text.insert(tk.END, "{}\n\n".format(cluster_movies))

    # Print outliers
    result_text.insert(tk.END, "Outliers:\n{}\n".format(outliers))
    result_text.insert(tk.END, "Number of outliers: {}\n".format(len(outliers)))
    result_text.insert(tk.END, "Number of final records: {}\n".format(sum))
    result_text.config(state=tk.DISABLED)


# Create main window
root = tk.Tk()
root.title("K-Means Clustering with Outlier Detection")

# File path entry
file_path_frame = tk.Frame(root)
file_path_frame.pack(pady=5)
file_path_label = tk.Label(file_path_frame, text="File Path:")
file_path_label.pack(side=tk.LEFT)
file_path_entry = tk.Entry(file_path_frame, width=50)
file_path_entry.pack(side=tk.LEFT, padx=5)
browse_button = tk.Button(file_path_frame, text="Browse", command=browse_file_path)
browse_button.pack(side=tk.LEFT)

# K entry
k_frame = tk.Frame(root)
k_frame.pack(pady=5)
k_label = tk.Label(k_frame, text="K:")
k_label.pack(side=tk.LEFT)
k_entry = tk.Entry(k_frame, width=10)
k_entry.pack(side=tk.LEFT, padx=5)

# Percentage entry
percentage_frame = tk.Frame(root)
percentage_frame.pack(pady=5)
percentage_label = tk.Label(percentage_frame, text="Percentage of Data to Use:")
percentage_label.pack(side=tk.LEFT)
percentage_entry = tk.Entry(percentage_frame, width=10)
percentage_entry.pack(side=tk.LEFT, padx=5)

# Start button
start_button = tk.Button(root, text="Start", command=start_process)
start_button.pack(pady=5)

# Result text
result_text = tk.Text(root, width=80, height=20, wrap=tk.WORD, state=tk.DISABLED)
result_text.pack(pady=5)

root.mainloop()
