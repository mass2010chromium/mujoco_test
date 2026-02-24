from dataclasses import dataclass
import os
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image
import torch

from transformers import AutoProcessor, AutoModel

import sam3
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import (
    visualize_frame_output
)

@dataclass
class GroundingResult:
    box: np.ndarray         # xyxy box in pixel space
    box_xywh: np.ndarray    # xywh box in normalized space (0-1)
    crop: np.ndarray        # RGB image array
    mask: np.ndarray        # Binary mask, 1 is included
    score: float            # Score measure of how confident this result is

    def to_dict(self):
        return {
            'box_xyxy': self.box.tolist(),   # TODO: does this actually help?
            'confidence': self.score
        }


class ObjectGrounder:
    def __init__(self, gpus_to_use=[0]):
        sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
        self.predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use, checkpoint_path=os.path.join(sam3_root, 'weights', 'sam3.pt'))
        self.clip_model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
        self.clip_processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        self.active_session_id = None
        self.video_data = None

    def _close_session(self):
        """
        Close the active inference session and free GPU resources.
        Not thread safe!
        """
        if self.active_session_id is not None:
            _ = predictor.handle_request(request=dict(
                type="close_session",
                session_id=self.active_session_id,
            ))
            self.active_session_id = None

    def _start_session(self, video_data: List[Image]):
        """
        Paper thin wrapper around SAM3 predictor.start_session()
        Closes the existing inference session, if any.
        Not thread safe!

        @param video_data:  List of PIL Images, forming a video.
                            Also saved to self.video_data.
        @return SAM3 response, with things like session ID.
                Session ID is saved to self.active_session_id
        """
        self._close_session()
        response = self.predictor.handle_request(request=dict(
            type="start_session",
            resource_path=video_data,
        ))
        self.active_session_id = response['session_id']
        self.video_data = video_data
        return response

    def predict_initial_masks(self, image: Image, classes: List[str], det_threshold=0.02, new_det_threshold=0.1):
        """
        Construct a set of initial object hypotheses based on a list of classes.

        Precondition: The video predictor session is started, with the appropriate video data.
        Likely also works with single image? (or one frame video?) Unsure.

        @param image:               PIL image to extract crops from.
        @param classes:             List of to semantic classes to query SAM 3 and extract masks for.
                                    Format should be: "a(n) <space separated object descriptor>"
        @param det_threshold:       Threshold for detections (default for SAM 3 0.5, for this function 0.1 to overdetect)
        @param new_det_threshold:   Threshold for new detections between frames? Unsure how this interacts with det_threshold

        @return list of masks, where each mask is a dict with the following fields:
            mask_id:    ID of the mask (str). Not necessarily indicative of the final mask object type
                        due to potential merging.

        """
        img_W, img_H = image.size

        # Save old detection thresholds, and restore them at the end
        save_det_threshold = self.predictor.model.score_threshold_detection
        save_new_det_threshold = self.predictor.model.new_det_thresh
        try:
            self.predictor.model.score_threshold_detection = det_threshold
            self.predictor.model.new_det_thresh = new_det_threshold

            # For each class, prompt SAM3 with the class.
            all_tracks = {}
            for class_id, text in enumerate(classes):
                frame_idx = 0  # add a text prompt on frame 0
                response = self.predictor.handle_request(request=dict(
                    type="add_prompt",
                    session_id=self.active_session_id,
                    frame_index=frame_idx,
                    text=text,
                ))
                sam_output = response["outputs"]
                obj_id = text.split(' ', 1)[1].replace(' ', '_')
                if obj_id.startswith('a_'):
                    obj_id = obj_id[2:]
                elif obj_id.startswith('an_'):
                    obj_id = obj_id[3:]

                for _id, box_xywh, mask, score in zip(
                    sam_output['out_obj_ids'],
                    sam_output['out_boxes_xywh'],
                    sam_output['out_binary_masks'],
                    sam_output['out_probs']
                ):
                    
                    score_vec = np.zeros(len(classes))
                    score_vec[class_id] = score
                    # Deduplicate greedily
                    tracks = list(all_tracks.values())
                    reject = False
                    for track in tracks:
                        other_mask = track['mask']
                        intersection = np.sum(mask * other_mask)
                        union = np.sum((mask + other_mask) != 0)
                        if intersection > 0 and intersection/union > 0.9:
                            if score > np.max(track['scores']):
                                del all_tracks[track['obj_id']]
                                # Instead of adding a new object, update the score for the existing object
                                score_vec += track['scores']
                            else:
                                track['scores'] += score_vec
                                reject = True
                                break   # Avoid adding score more than once
                    if reject:
                        continue

                    ## Extract xyxy bounding box and crop.
                    p1 = box_xywh[:2]
                    p2 = box_xywh[2:] + p1

                    # NOT matrix multiplication. Multiplies first column by img_W and second column by img_H
                    box_pixel = np.array([p1, p2]) * np.array([img_W, img_H])
                    crop = image.crop(box_pixel.flatten())

                    tmp_id = f"{obj_id}_{_id}"
                    all_tracks[tmp_id] = dict(
                        obj_id=tmp_id,
                        box=box_pixel,
                        box_xywh=box_xywh,
                        crop=crop,
                        mask=mask,
                        scores=score_vec    # Scores for each class
                    )
            return all_tracks
        finally:
            self.predictor.model.score_threshold_detection = save_det_threshold
            self.predictor.model.new_det_thresh = save_new_det_threshold

    def predict_masks_video(self, video_data: List[Image],
            classes: List[str], objects: List[str], use_clip=False) -> List[GroundingResult]:
        """
        Predict initial masks from video data.

        @param video_data:  List of PIL images
        @param classes:     Prompts for SAM, to generate detections
        @param objects:     Prompts for CLIP (appearance only, no location)
        """
        print("Grounding objects:")
        for i, cls in enumerate(classes):
            print(i, cls)
        self._start_session(video_data)

        # NOTE: for semantic (text) prompts, it seems like we don't need to reset_session every time.
        # If weird issues start cropping up consider moving this reset into the
        # predict_initial_masks function.
        _ = self.predictor.handle_request(request=dict(
            type="reset_session",
            session_id=self.active_session_id,
        ))

        image = video_data[0]
        all_tracks = self.predict_initial_masks(image, classes)
        print(len(all_tracks), "tracks.")
        print(len(objects), "objects.")

        # Construct a bipartite graph, with one side being all the (deduplicated) masks, and the other side
        # being the objects we want to match with. We will see a maximal matching where the edge weights
        # are the semantic similarity.

        G = nx.Graph()
        # Actually, I lied. the object nodes are just the object IDs (as in, order in the list)
        for obj_id, _ in enumerate(objects):
            G.add_node(obj_id, bipartite=0)

        # The mask nodes are labelled using their mangled names -- not necessarily matching their semantic content
        tracks = list(all_tracks.values())
        if use_clip:
            sam_weight = 0.5
            clip_weight = 0.5
            with torch.no_grad():
                # NOTE: CLIP is run for every mask individually, instead of batched.
                # Can we increase speed if they are all batched together?
                all_images = [track['crop'] for track in tracks]
                inputs = self.clip_processor(text=objects, images=all_images, padding="max_length", return_tensors="pt")
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                clip_scores = torch.sigmoid(logits_per_image) # these are the probabilities
        else:
            sam_weight = 1.0
            clip_weight = 0.0
            clip_scores = np.zeros((len(tracks), len(objects)))
        for i, track in enumerate(tracks):
            G.add_node(track['obj_id'], bipartite=1)
            for j, obj_ref in enumerate(objects):
                # CRITICAL: this `float()` forces the tensor object into a constant to be saved.
                weight = clip_weight*float(clip_scores[i][j]) + sam_weight*track['scores'][j]
                G.add_edge(track['obj_id'], j, weight=weight)

        # maxcardinality to force each requested object to have a match
        matching = nx.max_weight_matching(G, maxcardinality=True)
        # Sketch: Object labels are integers, masks are strings.
        # We force the mapping to be object -> mask
        matching = dict(x if (type(x[0]) == int) else x[::-1] for x in matching)
        print("Matching:", matching)

        # DEBUG PLOTTING
        output_maps = {
            'out_boxes_xywh': [],
            'out_probs': [],
            'out_obj_ids': [],
            'out_binary_masks': [],
        }
        for obj_id in range(len(objects)):
            if obj_id not in matching:
                continue
            track_name = matching[obj_id]
            obj = all_tracks[track_name]
            output_maps['out_boxes_xywh'].append(obj['box_xywh'])
            output_maps['out_probs'].append(G[obj_id][track_name]['weight'])
            output_maps['out_obj_ids'].append(obj_id)
            output_maps['out_binary_masks'].append(obj['mask'])
        visualize_frame_output(0, video_data, output_maps)
        plt.savefig('grounding_out.png')


        # NOTE: no way to reject matches.
        results = []
        for obj_id, _ in enumerate(objects):
            if obj_id not in matching:
                results.append(None)
                continue
            track_name = matching[obj_id]
            similarity = G[obj_id][track_name]['weight']
            track = all_tracks[track_name]
            results.append(GroundingResult(
                box=track['box'],
                box_xywh=track['box_xywh'],
                crop=track['crop'],
                mask=track['mask'],
                score=similarity
            ))
        return results


    def propagate_in_video(self):
        # we will just propagate from frame 0 to the end of the video
        outputs_per_frame = {}
        for response in self.predictor.handle_stream_request(request=dict(
            type="propagate_in_video",
            session_id=self.active_session_id,
        )):
            outputs_per_frame[response["frame_index"]] = response["outputs"]

        return outputs_per_frame


    def propagate_all_detections(self, detections, robot_pos=None):
        # Funny: Initialize data structures...
        _ = self.predictor.handle_request(request=dict(
            type="reset_session",
            session_id=self.active_session_id,
        ))
        _ = self.predictor.handle_request(request=dict(
            type="add_prompt",
            session_id=self.active_session_id,
            frame_index=0,
            text="nothing",
        ))
        # Can be slow, runs at 4FPS...
        # But experimentally I can skip every 5 frames and run it anyway :)
        # So maybe this can actually run at realtime or near realtime
        self.propagate_in_video()
        # TODO: finish implementation
