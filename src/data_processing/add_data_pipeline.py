from .pipeline_base import PipelineBase
import numpy as np
from ..utils import log_message, Segment, download_image, extract_protein_regions
from .WorkItems import WorkItem
from ..utils.clustering import perform_clustering_and_build_graph
from ..utils.resnet_classifier import extract_features
import json

class add_data_pipeline(PipelineBase):
    def __init__(self):
        super().__init__()
        self.task_name = 'add_data_pipeline'

    def execute(self, image_url):

        # Initialise workitem
        workitem = WorkItem()
        workitem.set_attribute('image_url', image_url)

        # Download image from Human Protein Atlas
        image = download_image(image_url=workitem.get_attribute('image_url'))
        image = np.array(image)
        workitem.set_attribute('image', image)

        # 1. Perform superpixel segmentation
        regions, images = extract_protein_regions(workitem.get_attribute('image_url'))
        workitem.set_attribute('regions', regions)
        workitem.set_attribute('images', images)

        # 2. Feature extraction
        feature_vectors = extract_features(workitem.get_attribute('images'))
        workitem.set_attribute('feature_vectors', feature_vectors)

        # 3. Clustering
        graph_dict = perform_clustering_and_build_graph(workitem.get_attribute('feature_vectors'))

        # Output worker result as json
        json_graph = json.dumps(graph_dict, indent=2)

        return json_graph
