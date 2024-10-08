from PIL import Image
from pathlib import Path
'''
extracts 2 subimages from a paged tiff and saves them as *_subimage_{x}.png. Second part overlays page 1 over page 0 and
saves as overlay_*.png.
'''
def extract_subimages_from_tiff(tiff_path):
    tiff_image = Image.open(tiff_path)

    # Check if the TIFF image contains multiple pages
    if 'n' in tiff_image.info:
        num_pages = tiff_image.n_frames
    else:
        num_pages = 2

    subimages = []

    # Iterate over each page/subimage in the TIFF image
    for page in range(num_pages):
        tiff_image.seek(page)
        subimage = tiff_image.copy()
        subimages.append(subimage)

    return subimages

# folder_path = './data/testing/'
#
# # Get a list of all TIFF files in the folder
# tiff_files = [file for file in os.listdir(folder_path) if file.endswith('.tif')]
#
# # Iterate over each TIFF file in the folder
# for tiff_file in tiff_files:
#     # Construct the full path to the TIFF file
#     tiff_path = os.path.join(folder_path, tiff_file)
#
#     # Extract subimages from the TIFF file
#     extracted_subimages = extract_subimages_from_tiff(tiff_path)
#
#     # Save each subimage as a separate file
#     for i, subimage in enumerate(extracted_subimages):
#         subimage.save(os.path.join(folder_path, f'{tiff_file}_subimage_{i}.png'))

# Define the paths to the folders
folder_a_path = Path("./data/subimages_0/")
folder_b_path = Path("./data/subimages_1/")

# Iterate over the images in folder A
for image_path in folder_a_path.glob("*.png"):
    # Open the image from folder A
    image_a = Image.open(image_path)

    # Get the corresponding image from folder B
    image_b_path = folder_b_path / (image_path.stem + ".png")
    if image_b_path.exists():
        image_b = Image.open(image_b_path)

        # Resize image B to match the size of image A
        image_b = image_b.resize(image_a.size)

        # Create a new image with 50% opacity
        image_b = image_b.convert('RGB')
        overlay = Image.blend(image_a, image_b, alpha=0.5)

        # Save the overlay image
        overlay.save("./data/testing/overlay/overlay_" + image_path.name)
    else:
        print(f"Image not found for {image_path.name} in folder B")