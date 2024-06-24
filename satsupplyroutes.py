import numpy as np
from PIL import Image
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sentinelhub import SHConfig, DataCollection, MimeType, CRS, BBox, SentinelHubRequest, bbox_to_dimensions, DownloadRequest
import matplotlib.pyplot as plt
from skimage.filters import sobel
from skimage.morphology import dilation, square
from skimage import feature, filters, morphology, transform
import tensorflow as tf
import tensorflow_hub as hub


class SentinelDataFetcher:
    def __init__(self, instance_id, client_id, client_secret, data_folder='./satimages'):
        """
        Initialize the data fetcher with Sentinel Hub credentials.
        """
        self.config = SHConfig()
        self.config.instance_id = instance_id
        self.config.sh_client_id = client_id
        self.config.sh_client_secret = client_secret
        self.data_folder = data_folder

    def calculate_bbox(self, longitude, latitude, buffer=0.02):
        """
        Calculate the bounding box around a central point with a buffer.
        """
        return BBox(bbox=[
            [longitude - buffer, latitude - buffer],
            [longitude + buffer, latitude + buffer]
        ], crs=CRS.WGS84)

    def fetch_and_save_data(self, longitude, latitude, date, resolution=10):
        """
        Fetch data for a given location and date.
        """
        time_interval = (date, date)  # Fetch data for a specific date
        bbox = self.calculate_bbox(longitude, latitude)
        size = bbox_to_dimensions(bbox, resolution=resolution)

        evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: ["B04", "B03", "B02"],
                output: { bands: 3 }
            };
        }
        function evaluatePixel(sample) {
            return [sample.B04, sample.B03, sample.B02];
        }
        """

        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L1C,
                    time_interval=time_interval,
                )
            ],
            responses=[
                SentinelHubRequest.output_response('default', MimeType.TIFF)
            ],
            bbox=bbox,
            size=size,
            config=self.config,
            data_folder=self.data_folder  # Make sure the data_folder is specified here
        )

        data = request.get_data(save_data=True)
        filename = f"{self.data_folder}/image_{date.replace('-', '')}.tiff"
        img = Image.fromarray(data[0])
        img.save(filename)
        print(f"Data for {date} fetched and saved at: {filename}")
        return filename

# Load a pre-trained DeepLab model from TensorFlow Hub
model_url = "https://tfhub.dev/google/deeplabv3/1"  # Example URL, check for the latest version
model = hub.load(model_url)

def run_model(image):
    """ Process an image with the TensorFlow model """
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    image_tensor = tf.image.resize(image_tensor, [256, 256])  # Resize if necessary
    image_tensor = tf.expand_dims(image_tensor, 0)  # Add batch dimension

    # Run model
    logits = model(image_tensor)
    predicted_image = tf.argmax(logits, axis=-1)
    predicted_image = predicted_image[0]  # Remove batch dimension

    return predicted_image.numpy()  # Convert to numpy array

def image_difference(image1_path, image2_path):
    # Load images
    img1 = Image.open(image1_path).convert('RGB')
    img2 = Image.open(image2_path).convert('RGB')

    # Convert images to arrays and normalize
    img1_array = np.array(img1) / 255.0
    img2_array = np.array(img2) / 255.0

    # Process images through the model
    img1_processed = run_model(img1_array)
    img2_processed = run_model(img2_array)

    # Calculate absolute difference of model outputs
    diff = np.abs(img1_processed.astype(int) - img2_processed.astype(int))

    # Use threshold to find significant changes
    threshold = np.where(diff > 0, 255, 0)

    # Optionally, apply morphological operations
    final_image = morphology.dilation(threshold, morphology.square(5))
    final_image = np.clip(final_image, 0, 255).astype(np.uint8)

    return final_image



# Configuration
instance_id = '7611b6f5-07cd-4b7e-8528-9a3a270da08d'
client_id = '44606ee6-b2f9-4ecc-8100-cc966b6ddf26'
client_secret = 'F0N89rMX0CpDgTezHaKbXhORYkaqY4ws'

# Initialize the data fetcher
fetcher = SentinelDataFetcher(instance_id, client_id, client_secret)

# Central coordinates for Kyiv
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


class ImageAnalysis:
    def __init__(self, image_path):
        """
        Initializes the ImageAnalysis class with the path to the image.
        """
        self.image_path = image_path
        self.data = self.load_image()

    def load_image(self):
        """
        Loads an image from the specified path and converts it to a numpy array.
        """
        img = Image.open(self.image_path)
        return np.array(img)

    def display_image(self):
        """
        Displays the image with a color map and color bar.
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(self.data, cmap='gray')
        plt.colorbar()
        plt.title('Visual Inspection of the Image')
        plt.show()

    def print_statistics(self):
        """
        Prints the basic statistics about the image data.
        """
        print(f"Data Type: {self.data.dtype}")
        print(f"Min Pixel Value: {self.data.min()}")
        print(f"Max Pixel Value: {self.data.max()}")
        print(f"Mean Pixel Value: {self.data.mean()}")
        print(f"Standard Deviation of Pixel Values: {self.data.std()}")

    def plot_histogram(self):
        """
        Plots a histogram of the pixel intensities in the image.
        """
        plt.figure()
        plt.hist(self.data.flatten(), bins=50, color='black')
        plt.title('Histogram of Pixel Intensities')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.show()

# Usage
image_analysis = ImageAnalysis('./satimages/diff_image.tiff')
image_analysis.display_image()
image_analysis.print_statistics()
image_analysis.plot_histogram()

