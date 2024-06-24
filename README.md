README.md
Sentinel Data Fetcher and Image Analysis Tool
This Python script is designed to fetch satellite images from the Sentinel Hub API, compare them using a pre-trained TensorFlow model, and analyze the resulting difference image. The script includes a class for fetching Sentinel data and another for analyzing images.

Requirements
To run this script, you will need the following Python packages:

numpy

pillow

datetime

matplotlib

sentinelhub

scikit-image

tensorflow

tensorflow_hub

You can install these packages using pip:

bash
Copy Code
pip install numpy pillow datetime matplotlib sentinelhub scikit-image tensorflow tensorflow_hub
SentinelDataFetcher Class
The SentinelDataFetcher class is responsible for fetching Sentinel satellite images for a given location and date. It requires Sentinel Hub credentials to access the data.

Initialization
python
Copy Code
SentinelDataFetcher(instance_id, client_id, client_secret, data_folder='./satimages')
instance_id: Your Sentinel Hub instance ID.

client_id: Your Sentinel Hub client ID.

client_secret: Your Sentinel Hub client secret.

data_folder: The folder where fetched images will be saved (default: './satimages').

Methods
calculate_bbox(longitude, latitude, buffer=0.02): Calculates the bounding box around a central point with a buffer.
fetch_and_save_data(longitude, latitude, date, resolution=10): Fetches data for a given location and date, and saves it to the specified folder.
ImageAnalysis Class
The ImageAnalysis class is used to analyze the difference image obtained from comparing two satellite images.

Initialization
python
Copy Code
ImageAnalysis(image_path)
image_path: The path to the image file to be analyzed.
Methods
load_image(): Loads the image from the specified path and converts it to a numpy array.

display_image(): Displays the image with a color map and color bar.

print_statistics(): Prints basic statistics about the image data.

plot_histogram(): Plots a histogram of the pixel intensities in the image.

Usage
Replace the instance_id, client_id, and client_secret with your Sentinel Hub credentials.

Set the longitude and latitude for the location you want to fetch data for.

Run the script to fetch images for the specified dates and analyze the difference image.

Example
python
Copy Code
# Configuration
instance_id = 'your-instance-id'
client_id = 'your-client-id'
client_secret = 'your-client-secret'

# Initialize the data fetcher
fetcher = SentinelDataFetcher(instance_id, client_id, client_secret)

# Central coordinates for the location of interest
longitude = 30.5234
latitude = 50.4501

# Dates for fetching data
yesterday = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
day_before_yesterday = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

# Fetch images for yesterday and the day before yesterday
image_path_yesterday = fetcher.fetch_and_save_data(longitude, latitude, yesterday)
image_path_day_before = fetcher.fetch_and_save_data(longitude, latitude, day_before_yesterday)

# Compare images
difference = image_difference(image_path_day_before, image_path_yesterday)
diff_img = Image.fromarray(difference)
diff_img.show()  # Display the difference image

# Optionally, save the difference image
diff_img.save(fetcher.data_folder + '/diff_image.tiff')

# Analyze the difference image
image_analysis = ImageAnalysis('./satimages/diff_image.tiff')
image_analysis.display_image()
image_analysis.print_statistics()
image_analysis.plot_histogram()
Notes
Make sure to handle your Sentinel Hub credentials securely and do not share them.

The script uses a pre-trained DeepLab model from TensorFlow Hub for image processing. You may need to update the model_url to the latest version or choose a different model.

The image_difference function calculates the absolute difference between two processed images and applies a threshold to highlight significant changes.

The ImageAnalysis class provides basic image analysis tools, including displaying the image, printing statistics, and plotting a histogram of pixel intensities.

License
This script is provided under the MIT License. Feel free to use, modify, and distribute it as you see fit.
