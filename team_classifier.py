'''
References: 
Tarek, A. (2024). Football_Analysis Team Analysis Code. [online] GitHub. 
Available at: https://github.com/abdullahtarek/football_analysis/blob/main/team_assigner/team_assigner.py [Accessed april 2nd 2025].

https://matplotlib.org/stable/gallery/mplot3d/scatter3d.html#sphx-glr-gallery-mplot3d-scatter3d-py accesed april 2nd 2025
https://medium.com/@fatimahk_26822/read-and-displaying-multiple-images-in-python-ac6f9be638ef accessed april 2nd 2025. 

'''

# Import packages for numerical operations, 3D scatter plots, K-means clustering, image processing, and file path handling 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2
import os 

class TeamAssigner:
    """
    Class to organise players into teams based on their shirt colour using K-means clustering. 

    Attributes:
    team_colours (array): store RGB colour values for the two teams. 
    player_team_dict (dictionary): maps the player_ids to their corresponding teams.  

    Methods:
    get_clustering_model(): Performs k_means clustering on images to find players shirt colour. 
    enhance_colour(): Enhances player colour to improve accuracy. 
    get_player_colour(): Find the players shirt colour. 
    assign_team_colour(): For all player_images clustered into two teams using K-means clustering
    get_player_team(): Return the team number based on the players_id either 1 or 2. 
    """
    def __init__(self):
        self.team_colours = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        """
        Get Team Shirt Colour For Each Player

        Args:
            image: Bounding box of players detected. 
        Outputs:
            Cluster of team colour values, separated from the background
        """
        
        # Reshape the image to 2D array
        image_2d = image.reshape(-1, 3)

        # Option 1: KMeans++ (Recommended) - better clustering for more consistent teams
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init="auto")

        # Option 2: Standard KMeans - uses random initialisation, more variance
        #kmeans = KMeans(n_clusters=2, init="random", n_init="auto") 
        
        # fit the clustering model on shirt colours for all players 
        kmeans.fit(image_2d)
        
        return kmeans

    def enhance_colour(self, colour):
        """
        Enhance the detected player shirt colour to make classification into two teams. 

        Args:
            colour: player recongised team colour
        Outputs:
            enhanced colour output for the classified team colours. 
        """
        # Strengthen the highest value and weaken smaller ones
        max_val = np.max(colour)
        modified_colour = np.array([
            val * 1.2 if val == max_val else val * 0.8
            for val in colour
        ])

        # Clip to valid RGB range
        return np.clip(modified_colour, 0, 255)

    def get_player_colour(self, image):
        """
        Obtain Player Shirt Colour

        Args:
            image: player image
        
        Outputs:
            Enhanced Colour for each of the players. 
        """
        
        # OPTION 1: Resize image to a fixed size (e.g., 64x64) for consistent input across players
        # This helps standardise clustering but may lose some detail.
        image_resized = cv2.resize(image, (64, 64))
        top_half_image = image_resized[:image_resized.shape[0] // 2, :]

        # OPTION 2 (currently used): Use original image size (may be inconsistent but retains detail)
        # Extract top half of the image containing the shirt)
        #top_half_image = image[:image.shape[0] // 2, :]
        
        # Get clustering model
        kmeans = self.get_clustering_model(top_half_image)
        
        # Get the cluster labels for each pixel
        labels = kmeans.labels_

        # reshape the labels into the orginal image shape
        clustered_image = labels.reshape(top_half_image.shape[:2])

        # Determine the player's dominant cluster by comparing to the corner pixel RGB values usually grass green
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1],
                           clustered_image[-1, 0], clustered_image[-1, -1]]

        # extract the player shirt colour from background. 
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        # Get the dominant colour for the player's cluster
        player_colour = np.flip(kmeans.cluster_centers_[player_cluster]).astype(int)

        # EARLY RETURN OPTION:
        # If you do NOT want to use enhancement, return player_colour directly:
        # return player_colour

        # ENHANCEMENT OPTION:
        # Strengthen primary shirt colour (e.g., make red shirts more "red")
        return self.enhance_colour(player_colour)

    def assign_team_colour(self, player_images):
        """
        Assign two team colours based on shirt colour

        Args:
            player_images: Extracted Player 
        Outputs:
            Two teams classifed using k-means clustering. 
        """

        # determine player shirt colours
        player_colours = [self.get_player_colour(image) for image in player_images]
        
        # Option 1: KMeans++ (Recommended) - better clustering for more consistent teams
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init="auto")

        # Option 2: Standard KMeans - uses random initialisation, more variance
        #kmeans = KMeans(n_clusters=2, init="random", n_init="auto")

        kmeans.fit(player_colours)

        # assign players into two teams.
        self.kmeans = kmeans
        self.team_colours[1] = np.flip(kmeans.cluster_centers_[0].astype(int))
        self.team_colours[2] = np.flip(kmeans.cluster_centers_[1].astype(int))

    def get_player_team(self, image, player_id):
        """
        Assign players into team 1 and team 2. 

        Args:
            image: uploaded image of a football match 
            player_id: id of that players corresponding bounding box
        Outputs:
            Team_id for each player. 
        """
        # If already assigned, return the team
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # if player is not assigned extract their shirt 
        player_colour = self.get_player_colour(image)

        # based on shirt colour assign to corresponding team id + 1 to make it team 1 and team 2. 
        team_id = self.kmeans.predict(player_colour.reshape(1, -1))[0] + 1

        # store corresponding team_id for player id. 
        self.player_team_dict[player_id] = team_id

        return team_id
        
def load_images_from_folder(folder_path):
    """
        Extracts all images and corresponding files names from folder. 

        Args:
            folder_patch: folder path chosen by user. 
        Outputs:
            images: RGB images from folder. 
            filenames: corresponding filenames 
    """
    
    # initialise to arrays for images and filenames. 
    images = []
    filenames = []
    
    # for each filename in folder store the RGB image and corresponding file name and append to arrays. 
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        images.append(img)
        filenames.append(filename)
    
    # return images of player's and corresponding filenams for use in TeamAssigner
    return images, filenames

def visualise_results(player_images, filenames, player_teams, player_colours, team_colours):
    """
        Visualise k-means cluster 3D scatter plot and team classification results 

        Args:
            player_images: array of all player images 
            filenames: array of corresponding filenames 
            player_teams: Team 1 or Team 2 for each player
            player_colours: RGB shirt colour for each player
            team_colours: RGB team colours. 
        Outputs:
            a plot of players showing their predicted shirt colours, which team they have been assigned, compared to the predicted team colours. 
    """

    num_players = len(player_images)
    
    # Create a grid: 2 rows (Players & Shirt Colours) + 1 row for Team Colours
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

    # show plot with title. 
    plt.suptitle("K-Means Player Team Classification Results", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def scatter_plot_kmeans(player_colours, centroids):
    """
        Produces a 3D scatter plot representing the K-means clustered RGB colour values. 

        Args:
            player_colours: RGB team colour values. 
            centroids: RGB value of the centroids from K-means. 
        Outputs:
    """

    # add a 3d plot onto a figure. 
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    print(f"player_colours length: {len(player_colours)}")

    # for each colour in player colours plot onto 3D scatterplot. 
    for _, colour in player_colours.items(): 
        red, green, blue = colour  

        #print(f"{filename}: R={red}, G={green}, B={blue}")

        # plot red green and blue on x,y an z axis with the colour being normalised between 0 and 1. 
        ax.scatter(red, green, blue, color=np.array(colour) / 255)

    # plot the centroids. 
    for _, centroid in centroids.items():
        np_centroid = np.array(centroid)

        # plot the centroids using coordinates and the colour is once again normalised. 
        ax.scatter(*np_centroid, color=np_centroid / 255, marker='X', s=200, edgecolor='black')
        print(np_centroid)

    # tick labels are in increments of 50 up to 255.
    ax.set_xticks([0, 50, 100, 150, 200, 255])
    ax.set_yticks([0, 50, 100, 150, 200, 255])
    ax.set_zticks([0, 50, 100, 150, 200, 255])

    # display labels and show plot
    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_title("Player Shirt Colours - K-Means Clusters")
    plt.show()


def main(folder_path):
    """
        runs the team extractor on a folder of images 

        Args:
            folder_path: specified folder by user. 
        Outputs:
            two plots above and RGB colour values and teams assigned for each player and teams. 
    """

    # initialise team assigner and load images and filenames. 
    team_assigner = TeamAssigner()
    player_images, filenames = load_images_from_folder(folder_path)
    
    # get team colours. 
    team_assigner.assign_team_colour(player_images)

    #print team information. 
    print("\n=== Team Colours Identified ===")
    print(f"Team 1 Colour: {team_assigner.team_colours[1]}")
    print(f"Team 2 Colour: {team_assigner.team_colours[2]}")
    print("==============================\n")

    # empty player_teams and player_colours dictionary. 
    player_teams = {}
    player_colours = {}

    # for each image and filename get player shirt RGB value, assigned teams 
    for idx, (image, filename) in enumerate(zip(player_images, filenames)):
        
        # get player colour
        player_colour = np.flip(team_assigner.get_player_colour(image)).astype(int)
        
        # get corresponding team_id
        team_id = team_assigner.get_player_team(image, idx)
        
        # append team_id and player colour to corresponding filenames. 
        player_teams[filename] = team_id
        player_colours[filename] = player_colour
        
        print(f"Player {filename} - Extracted Colour: {player_colour} - Assigned to Team {team_id}")
        
    # Visualise results
    visualise_results(player_images, filenames, player_teams, player_colours, team_assigner.team_colours)
    scatter_plot_kmeans(player_colours, team_assigner.team_colours)

if __name__ == "__main__":

    # Define the base directory containing all extracted player folders
    base_folder = "dataset/extracted_players/"

    # Prompt the user to choose how they want to run the program
    print("Would you like to:")
    print("1. Process all folders in the dataset")
    print("2. Process a single specified folder")
    choice = input("Enter 1 or 2: ")
    
    # Option 1: Iterate through every folder within the base directory
    if choice == "1":
        for folder_name in os.listdir(base_folder):
            folder_path = os.path.join(base_folder, folder_name)

            # Check if the item is a directory (i.e. a valid folder)
            if os.path.isdir(folder_path):
                print(f"\n\n=== Processing Folder: {folder_name} ===")
                main(folder_path)
    
    # Option 2: Hardcoded path to a specific folder (can be adjusted or extended to take input)
    elif choice == "2":    
        folder_path = "dataset/extracted_players/red_team_vs_green_team"
        print(f"\n=== Processing Folder: {folder_path} ===")
        main(folder_path)
    
    else:
        # Catch any invalid user inputs
        print("Invalid input. Please enter 1 or 2.")
