'''
References: 
Tarek, A. (2024). Football_Analysis Team Analysis Code. [online] GitHub. 
Available at: https://github.com/abdullahtarek/football_analysis/blob/main/team_assigner/team_assigner.py [Accessed 11 Dec. 2024].

https://matplotlib.org/stable/gallery/mplot3d/scatter3d.html#sphx-glr-gallery-mplot3d-scatter3d-py accesed april 2nd 2025
https://medium.com/@fatimahk_26822/read-and-displaying-multiple-images-in-python-ac6f9be638ef accessed april 2nd 2025. 

'''


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2
import os 

class TeamAssigner:
    def __init__(self):
        self.team_colours = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)
        return kmeans

    def enhance_colour(self, colour):
        max_val = np.max(colour)
        modified_colour = np.array([
            val * 1.2 if val == max_val else val * 0.8
            for val in colour
        ])
        return np.clip(modified_colour, 0, 255)

    def get_player_colour(self, image):
        image_resized = cv2.resize(image, (64, 64))
        top_half_image = image_resized[:image_resized.shape[0] // 2, :]

        kmeans = self.get_clustering_model(top_half_image)
        labels = kmeans.labels_
        clustered_image = labels.reshape(top_half_image.shape[:2])

        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1],
                           clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_colour = np.flip(kmeans.cluster_centers_[player_cluster]).astype(int)
        return self.enhance_colour(player_colour)

    def assign_team_colour(self, player_images):
        player_colours = [self.get_player_colour(image) for image in player_images]
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init="auto")
        kmeans.fit(player_colours)

        self.kmeans = kmeans
        self.team_colours[1] = np.flip(kmeans.cluster_centers_[0].astype(int))
        self.team_colours[2] = np.flip(kmeans.cluster_centers_[1].astype(int))

    def get_player_team(self, image, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_colour = self.get_player_colour(image)
        team_id = self.kmeans.predict(player_colour.reshape(1, -1))[0] + 1

        self.player_team_dict[player_id] = team_id
        return team_id
        
def load_images_from_folder(folder_path):
    images = []
    filenames = []
    
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        images.append(img)
        filenames.append(filename)
    
    return images, filenames

def visualise_results(player_images, filenames, player_teams, player_colours, team_colours):
    num_players = len(player_images)
    
    # Create a grid: 2 rows (Players & Shirt Colors) + 1 row for Team Colors
    _, axes = plt.subplots(2, num_players + 2, figsize=(15, 4))  
    print(f"PLAYER Len: {len(player_images)}")
    
    # Display player images with labels (First row)
    for i, (img, filename) in enumerate(zip(player_images, filenames)):
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"T{player_teams[filename]}")
        axes[0, i].axis("off")

        # Display extracted shirt colours (Second row)
        colour_patch = np.full((50, 50, 3), player_colours[filename], dtype=np.uint8)
        axes[1, i].imshow(colour_patch)
        axes[1, i].axis("off")

    # Show the clustered team colours (Extra columns)
    for j, (team_id, colour) in enumerate(team_colours.items()):
        colour_patch = np.full((50, 50, 3), colour, dtype=np.uint8)
        axes[0, num_players + j].imshow(colour_patch)
        axes[0, num_players + j].set_title(f"T{team_id}")
        axes[0, num_players + j].axis("off")
        
        # Add an empty subplot below to align structure
        axes[1, num_players + j].axis("off")

    plt.tight_layout()
    plt.show()

def scatter_plot_kmeans(player_colours, centroids):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    print(f"player_colours length: {len(player_colours)}")

    for _, colour in player_colours.items(): 
        red, green, blue = colour  

        #print(f"{filename}: R={red}, G={green}, B={blue}")
        ax.scatter(red, green, blue, color=np.array(colour) / 255)

    for i, centroid in centroids.items():
        normalised_centroid = np.array(centroid)
        ax.scatter(*normalised_centroid, color=normalised_centroid / 255, marker='X', s=200, edgecolor='black', label=f"Centroid {i}")
        print(normalised_centroid)


    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_title("Player Shirt Colors - KMeans Clusters")
    plt.show()


def main(folder_path):
    team_assigner = TeamAssigner()
    player_images, filenames = load_images_from_folder(folder_path)
    
    team_assigner.assign_team_colour(player_images)

    print("\n=== Team Colors Identified ===")
    print(f"Team 1 Color: {team_assigner.team_colours[1]}")
    print(f"Team 2 Color: {team_assigner.team_colours[2]}")
    print("==============================\n")

    player_teams = {}
    player_colours = {}

    for idx, (image, filename) in enumerate(zip(player_images, filenames)):
        player_colour = np.flip(team_assigner.get_player_colour(image)).astype(int)
        team_id = team_assigner.get_player_team(image, idx)
        player_teams[filename] = team_id
        player_colours[filename] = player_colour
        print(f"Player {filename} - Extracted Color: {player_colour} - Assigned to Team {team_id}")
        
    # Visualise results
    visualise_results(player_images, filenames, player_teams, player_colours, team_assigner.team_colours)
    scatter_plot_kmeans(player_colours, team_assigner.team_colours)

if __name__ == "__main__":
    folder_path = "dataset/extracted_players/light_blue_team_vs_blue_team"  
    main(folder_path)
