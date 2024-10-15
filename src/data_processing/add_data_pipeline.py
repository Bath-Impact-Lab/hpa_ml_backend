from sklearn.decomposition import PCA

from .pipeline_base import PipelineBase
import numpy as np
from ..utils import download_image, extract_protein_regions
from ..utils.clustering import perform_clustering_and_build_graph
from ..utils.resnet_classifier import extract_features
import json

from ..workers.WorkItems import WorkItem


class AddDataPipeline(PipelineBase):
    def __init__(self):
        super().__init__()
        self.task_name = 'add_data'

    def execute(self, image_url, protein, tissue_type):
        body = {
            'image_url': image_url,
            'protein': protein,
            'tissue_type': tissue_type
        }


        workitem = WorkItem(body=body)



        # Download image from Human Protein Atlas
        image = download_image(image_url=workitem.get_attribute('image_url'))
        image = np.array(image)
        workitem.set_attribute('image', image)

        # 1. Perform protein region extraction
        centers, images = extract_protein_regions(workitem.get_attribute('image'))
        # if len(images) > 20:
        #     images = images[:20]

        workitem.set_attribute('centers', centers)
        workitem.set_attribute('images', images)


        # 2. Feature extraction
        feature_vectors = extract_features(workitem.get_attribute('images'))
        workitem.set_attribute('feature_vectors', feature_vectors)


        # 3. Clustering
        graph_dict = perform_clustering_and_build_graph(workitem.get_attribute('feature_vectors'), image_url=workitem.get_attribute('image_url'), image_positions=workitem.get_attribute('centers'), protein=workitem.get_attribute('protein'), tissue_type=workitem.get_attribute('tissue_type'))

        # Output worker result as json
        json_graph = json.dumps(graph_dict, indent=2)

        return json_graph

