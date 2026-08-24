"""Classificacao de times a partir dos uniformes dos jogadores."""

from typing import Generator, Iterable, List, TypeVar

import numpy as np
import supervision as sv
import torch
import umap
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel

V = TypeVar("V")

SIGLIP_MODEL_PATH = "google/siglip-base-patch16-224"


def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[List[V], None, None]:
    """Divide uma sequencia em lotes.

    Args:
        sequence: Sequencia de entrada.
        batch_size: Tamanho desejado de cada lote.

    Yields:
        Lotes com ate `batch_size` elementos.
    """

    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class TeamClassifier:
    """Classifica jogadores em dois times por similaridade visual.

    O classificador usa SigLIP para extrair embeddings dos crops de jogadores,
    reduz a dimensionalidade com UMAP e agrupa os uniformes com KMeans.

    Args:
        device: Dispositivo de execucao do modelo de embeddings.
        batch_size: Quantidade de crops processados por lote.
    """

    def __init__(self, device: str = "cpu", batch_size: int = 32) -> None:
        self.device = device
        self.batch_size = batch_size
        self.features_model = SiglipVisionModel.from_pretrained(
            SIGLIP_MODEL_PATH
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=2)

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """Extrai embeddings visuais dos crops de jogadores.

        Args:
            crops: Lista de recortes em formato OpenCV.

        Returns:
            Matriz NumPy com os embeddings extraidos.
        """

        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops, self.batch_size)
        data = []
        with torch.no_grad():
            for batch in tqdm(batches, desc="Embedding extraction"):
                inputs = self.processor(images=batch, return_tensors="pt").to(
                    self.device
                )
                outputs = self.features_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        """Ajusta o agrupamento de times usando crops amostrados do video.

        Args:
            crops: Recortes de jogadores usados para calibrar os dois clusters.
        """

        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """Prediz o time de cada crop informado.

        Args:
            crops: Recortes de jogadores a classificar.

        Returns:
            Array com IDs de cluster, normalmente `0` ou `1`.
        """

        if len(crops) == 0:
            return np.array([])

        data = self.extract_features(crops)
        projections = self.reducer.transform(data)
        return self.cluster_model.predict(projections)
