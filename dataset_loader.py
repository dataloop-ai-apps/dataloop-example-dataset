import logging
import os
import requests
import zipfile
import json
import dtlpy as dl
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger('dataloop-example-dataset')


class DatasetExample(dl.BaseServiceRunner):
    """
    A class to handle upload of example dataset to Dataloop platform.
    """

    def __init__(self):
        """
        Initialize the dataset downloader and download the zip.
        """
        logger.info('Downloading zip file...')
        self.url = 'https://storage.googleapis.com/model-mgmt-snapshots/datasets-clustering-demo/export.zip'
        self.dir = os.getcwd()
        self.zip_dir = os.path.join(self.dir, 'export.zip')

        # Download the zip file
        response = requests.get(self.url)
        if response.status_code == 200:
            with open(self.zip_dir, 'wb') as f:
                f.write(response.content)
        else:
            logger.error(f'Failed to download the file. Status code: {response.status_code}')
            return

        # Extract the zip file
        with zipfile.ZipFile(self.zip_dir, 'r') as zip_ref:
            zip_ref.extractall(self.dir)
        logger.info('Zip file downloaded and extracted.')

    def upload_dataset(self, dataset: dl.Dataset, source: str):
        """
        Uploads the dataset to Dataloop platform, including items, annotations and feature vectors.

        :param dataset: The Dataloop dataset object where the data will be uploaded.
        """
        logger.info('Uploading dataset...')
        local_path = os.path.join(self.dir, 'export/items/')
        json_path = os.path.join(self.dir, 'export/json/')
        dataset.items.upload(local_path=local_path, local_annotations_path=json_path, item_metadata=dl.ExportMetadata.FROM_JSON)

        # Setup dataset recipe and ontology
        recipe = dataset.recipes.list()[0]
        ontology = recipe.ontologies.list()[0]
        ontology.add_labels(label_list=['zebra', 'cat', 'elephant', 'bird', 'tiger', 'snake', 'bat'])

        # Handle feature set
        feature_set = self.ensure_feature_set(dataset)

        # Upload features
        vectors_file = os.path.join(self.dir, 'export/vectors/vectors.json')
        with open(vectors_file, 'r') as f:
            vectors = json.load(f)

        with ThreadPoolExecutor(max_workers=10) as executor:
            for key, value in vectors.items():
                executor.submit(self.create_feature, key, value, dataset, feature_set)

    def ensure_feature_set(self, dataset):
        """
        Ensures that the feature set exists or creates a new one if not found.

        :param dataset: The dataset where the feature set is to be managed.
        """
        try:
            feature_set = dataset.project.feature_sets.get(feature_set_name='clip-feature-set')
            logger.info(f'Feature Set found! Name: {feature_set.name}, ID: {feature_set.id}')
        except dl.exceptions.NotFound:
            logger.info('Feature Set not found, creating...')
            feature_set = dataset.project.feature_sets.create(
                name='clip-feature-set',
                entity_type=dl.FeatureEntityType.ITEM,
                project_id=dataset.project.id,
                set_type='clip',
                size=512
            )
        return feature_set

    @staticmethod
    def create_feature(key, value, dataset, feature_set):
        """
        Creates a feature for a given item.

        :param key: The key identifying the item.
        :param value: The feature value to be added.
        :param dataset: The dataset containing the item.
        :param feature_set: The feature set to which the feature will be added.
        """
        item = dataset.items.get(filepath=key)
        feature_set.features.create(entity=item, value=value)
