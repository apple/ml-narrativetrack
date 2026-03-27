#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import os
import torch
import clip
from PIL import Image
import torchreid
import numpy as np
import face_recognition
import torchvision.transforms as T
from transformers import Owlv2Processor, Owlv2ForObjectDetection

from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.config import get_cfg

class EntityClustering:
    def __init__(self, sim_threshold=0.75):
        self.entity_clusters = []  # list of sets of person_ids
        self.entity_embeddings = []  # list of np.array embeddings
        self.sim_threshold = sim_threshold
    
    def add_person(self, person_id, person_embedding):
        """
        Add a new person_id with its embedding to the clustering.
        
        Returns: assigned_entity_idx (int), is_new_entity (bool)
        """
        if not self.entity_clusters:
            # First entity
            self.entity_clusters.append({person_id})
            self.entity_embeddings.append(person_embedding)
            return 0, True
        
        # Compare to existing entities
        sims = []
        for emb in self.entity_embeddings:
            sim = np.dot(person_embedding, emb.T)
            sims.append(sim)
        
        # Find best match
        max_sim = max(sims)
        best_idx = sims.index(max_sim)
        
        if max_sim >= self.sim_threshold:
            # Match → add to existing entity
            self.entity_clusters[best_idx].add(person_id)
            # Update entity embedding (average with new embedding)
            old_emb = self.entity_embeddings[best_idx]
            new_emb = (old_emb * len(self.entity_clusters[best_idx]) + person_embedding) / (len(self.entity_clusters[best_idx]) + 1)
            new_emb /= np.linalg.norm(new_emb)
            self.entity_embeddings[best_idx] = new_emb
            return best_idx, False
        else:
            # No match → create new entity
            self.entity_clusters.append({person_id})
            self.entity_embeddings.append(person_embedding)
            return len(self.entity_clusters) - 1, True
    
    def get_entity_clusters(self):
        return self.entity_clusters
        
def load_fastreid_model():
    # Load pretrained OSNet_x1_0 model (from MSMT17 → good generalization)
    model = torchreid.models.build_model(
        name='osnet_x1_0',
        num_classes=1000,
        pretrained=True
    )

    model.eval()
    model = model.to('cuda')
    return model

def load_clip_model():
    model, preprocess = clip.load("ViT-L/14", device="cpu")
    return model.eval(), preprocess

def compute_clip_embedding(model, preprocess, image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    with torch.no_grad():
        emb = model.encode_image(image).squeeze(0)
        emb = emb / emb.norm()
    return emb.cpu().numpy()

def preprocess():
    transform_reid = T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform_reid

def get_detector(args, device):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
    cfg.MODEL.DEVICE = device
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    return DefaultPredictor(cfg)

def get_owlv2_detector(args, device):
    # Load OWLv2 (OWL-ViT) model
    if args.owl_model_type == "base":
        owl_model_name = "google/owlv2-base-patch16-ensemble" 
    else:
        owl_model_name = "google/owlv2-large-patch14-ensemble"
    owl_processor = Owlv2Processor.from_pretrained(owl_model_name, local_files_only=False)
    owl_model = Owlv2ForObjectDetection.from_pretrained(owl_model_name).eval().to(device)
    return owl_processor, owl_model

def detect_person(detector, frame):
    with torch.no_grad():
        results = detector(frame)
    mask = results["instances"].pred_classes.cpu() == 0
    boxes = results["instances"].pred_boxes.tensor[mask].cpu()
    confs = results["instances"].scores.cpu().numpy() 
    return boxes, confs

def detect_person_owlv2(owl_processor, owl_model, frame, text_queries=["a photo of person"], threshold=0.27):
    if isinstance(frame, np.ndarray):
        image = Image.fromarray(frame)
    else:
        image = frame

    inputs = owl_processor(text=text_queries, images=image, return_tensors="pt").to(owl_model.device)
    with torch.no_grad():
        outputs = owl_model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])  # (H, W)
    results = owl_processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=threshold)[0]

    boxes, scores, labels = results["boxes"], results["scores"], results["labels"]
    # for box, score, label in zip(boxes, scores, labels):
    #     box = [round(i, 2) for i in box.tolist()]
    #     print(f"Detected {text_queries[label]} with confidence {round(score.item(), 3)} at location {box}")
    boxes = boxes.cpu()
    scores = scores.cpu().numpy()
    return boxes, scores

def build_face_db(ref_dir, video_id, face_dir):
    dir_path = os.path.join(ref_dir, video_id)
    encodings = []
    names = []
    for name in os.listdir(dir_path):
        if name.endswith(".png") or name.endswith(".jpg"):
            img_path = os.path.join(dir_path, name)
            image = face_recognition.load_image_file(img_path)
            enc = face_recognition.face_encodings(image)
            if enc:
                encodings.append(enc[0])
                names.append(os.path.splitext(name)[0])
        os.makedirs(os.path.join(face_dir, name.split(".")[0]), exist_ok=True)
    return {"encodings": encodings, "names": names}

