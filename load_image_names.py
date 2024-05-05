def load_image_names(filename):
    with open(filename, 'r') as file:
        image_names = file.read().split()
    return image_names


image_names = load_image_names('test.txt')

scores_indices = [(0, 34), (0, 56), (0, 88), (0, 88), (0, 88),
                  (0, 88), (0, 118), (0, 186), (0, 187), (0, 226)]

selected_images = [image_names[idx] for _, idx in scores_indices]

print(selected_images)
