def visualise_results(player_images, filenames, player_teams, player_colours, team_colours):
    num_players = len(player_images)
    
    # Create a grid: 2 rows (Players & Shirt Colors) + 1 row for Team Colors
    _, axes = plt.subplots(2, num_players + 2, figsize=(15, 4))  

    # Display player images with labels (First row)
    for i, (img, filename) in enumerate(zip(player_images, filenames)):
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"T{player_teams[filename]}")
        axes[0, i].axis("off")