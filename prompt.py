#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

GEMINI_ACTION_RECOG_PROMPT = """You are an expert video understanding model.

You will be shown a video clip that contains one clearly highlighted person using green bounding boxes. This person is the main subject of interest.

### Your Task:
You must describe what the highlighted person is doing **only** when they are clearly tracked by a visible bounding box. **Do not describe any actions** when bounding boxes are missing.

1. Describe the highlighted person’s actions **strictly within the tracked green bounding box regions**.
2. Identify fine-grained and meaningful human actions (e.g., "walk", "use phone", "stand", "wave") only when there is a person highlighted by the bounding box.
3. If the person highlighted by bounding boxe begins a new activity or changes behavior, describe the transition.
4. Determine whether the person undergoes a **significant change** in actions (e.g., switching from passive to active behavior or starting a new interaction).
5. Indicate whether the person’s actions are **clearly distinguishable** from the actions of others not highlighted by the bounding box.
6. Indicate whether a **similar-looking entity** is present in the same scene (e.g., someone with similar outfit or hairstyle).

### Important Notes:
- Do not describe non-highlighted people or the background.
- Skip all segments without bounding boxes. Do not infer, guess, or assume what happens during gaps in tracking.
- Focus **only** on observable actions in the bounding box, and do not infer intentions or emotions.
- For similar-looking entities, do not describe them, but just state if one is visibly present.
- For action distinguishability, do not explain what others are doing, but only assess whether the highlighted person’s action is visually distinct.

### Output Format (JSON):
Return a JSON object with the following fields:
- `"significant_transition"`: boolean — does the person change activity significantly?
- `"action_distinguishable"`: boolean — are their actions visually distinct from others?
- `"similar_looking"`: boolean — is there another person who looks visually similar?
- `"justification"`: object — short natural language justifications for each boolean.
- `"steps"`: list of action steps during bounding box regions only. Each item must include:
  - `"step"`: a short sentence describing what the person is doing (in that segment only).

Each justification should be concise (1–2 sentences), and based only on visible behavior **during valid bounding box segments**.

### Example Output (do not include markdown or backticks):
{
  "significant_transition": true or false,
  "action_distinguishable": true or false,
  "similar_looking": true or false,
  "justification": {
    "significant_transition": "reason for significant transition",
    "action_distinguishable": "reason for action distinguishability",
    "similar_looking": "reason for similar looking"
  },
  "steps": [
    {"step": "standing and looking around"},
    {"step": "starts walking across the room"}
  ]
}

Return only the raw JSON object — do NOT include any commentary, markdown, or formatting outside of it.
"""

GEMINI_SINGLE_ACTION_RECOG_PROMPT = """You are an expert video understanding model.

You will be shown a short video clip where a single person is clearly highlighted using a green bounding box. This person is the main subject of interest.

### Your Task:
Your goal is to extract the following information strictly based on visual evidence inside the green bounding box and the visible background:

1. Action — one fine-grained action the person is performing.
2. Outfit — describe what the person is wearing (clothing type, color, accessories).
3. Scene — briefly describe the background environment or setting (e.g., "kitchen", "forest trail", "office room").

If there is no bounding box visible throughout the video, return the message `"INVALID"` for all three fields.

### Requirements:
- Focus only on the person inside the green bounding box.
- Use precise and visually grounded descriptions.
- If the person is interacting with a visible item or person inside the bounding box, name the item or describe the outfit of the person specifically — do not use vague terms like “object” or “item”. (e.g., say “raising a suitcase” instead of “raising an object”).
- Please focus on observable behavior, not inferred intentions or emotions. 
- If the person transitions between multiple actions, pick the most dominant or meaningful action visible in the clip.

### Output Format (JSON):
Return exactly one of the following:

#### If bounding box is visible:
{
  "action": "cutting vegetables",
  "outfit": "blue short-sleeve shirt and white apron",
  "scene": "modern kitchen"
}

#### If no bounding box is visible:
{
  "action": "INVALID",
  "outfit": "INVALID",
  "scene": "INVALID"
}

Return only the raw JSON object — do NOT include any commentary, markdown, or explanation.
"""

GEMINI_ENTIRE_RECOG_PROMPT = """You are an expert video understanding assistant.

You will be given a video that includes the target entity consistently highlighted by green bounding boxes, along with summarized information about action, outfit, and scene changes involving a target entity.
Your task is to generate structured metadata based on the following criteria:
1. Determine whether the target entity shows significant action transitions — only based on the provided action change description.
2. Determine whether the target entity shows significant outfit transitions — only based on the provided outfit change description.
3. Determine whether the scene changes significantly significantly involving target entity — only based on the provided outfit change description.
4. Determine whether any similar-looking entity (i.e., someone with similar outfit) appears or not in the video, based on the provided video.
5. Describe the single action and outfit (e.g., clothes, color, accessories) of the other entities (not describe the changes) that are not highlighted by the bounding box, based on the provided video.
6. Describe wehther the target entity shows over three action transitions (e.g., talking -> walking -> talking -> crying) — only based on the provided action change description.
7. Describe whether the target entity shows over three outfit transitions (e.g., blue t-shirt -> white t-shirt -> pink dress -> black coat) — only based on the provided outfit change description.
8. Describe whether the target entity shows over three scene transitions (e.g., church -> stadium -> park -> indoor room) — only based on the provided scene change description.

In each case, return a binary decision as `true` or `false`, and provide a clear justification.
If there are only a single element in the changes, you should return `false` for the corresponding field.
If a similar-looking entity is present, also describe their outfit and actions in the justification.

Here are some examples of target entity information and generated structured metadata:
[Example 1] Target Entity Information in the video:
- Action Changes: ["sitting", "singing", "walking", "singing"]
- Outfit Changes: ["red shirt", "red stripped shirt", "red shirt", "red shirt"]
- Scene Changes: ["park", "bench", "park", "park"]

[Example 1] Output Metadata:
```json
{{
  "significant_action_transition": true,
  "significant_scene_transition": false,
  "significant_outfit_transition": false,
  "similar_looking_existence": true,
  "justification": {{
    "significant_action_transition": "The target entity changes actions from sitting on the bench to singing, walking, and singing.",
    "significant_scene_transition": "The target entity remains in the park and bench, where bench might be located in the park.",
    "significant_outfit_transition": "The target entity wears a red shirt and never changes outfits.",
    "similar_looking_existence": "There are other entity wearing a red t-shirts and blue pants in the video."
  }},
  "other_entities": [
    {{
      "actions": "talking to someone sitting on the bench",
      "outfits": "red t-shirts and blue pants"
    }}, 
    {{
      "actions": "walking around the bench",
      "outfits": "yellow hat and green jacket"
    }}, 
  ],
  "over_three_action_changes": true,
  "over_three_outfit_changes": false,
  "over_three_scene_changes": false
}}
---
[Example 2] Target Entity Information in the Video:
- Action Changes: ["talking"]
- Outfit Changes: ["red shirt, black jeans, blue hat"]
- Scene Changes: ["beach"]

[Example 2] Output Metadata:
```json
{{
  "significant_action_transition": false,
  "significant_scene_transition": false,
  "significant_outfit_transition": false,
  "similar_looking_existence": false,
  "justification": {{
    "significant_action_transition": "The target entity only appears during one segment, where the given inforamation has one action ("talking").",
    "significant_scene_transition": "The target entity only appears during one segment, where the given inforamation has one scene ("beach").",
    "significant_outfit_transition": ""The target entity only appears during one segment, where the given inforamation has one outfit ("red shirt, black jeans, and blue hat").",
    "similar_looking_existence": "There are no entity wearing a similar outfits to red shirt, black jeans and blue hat in the video."
  }},
  "other_entities": [
    {{
      "actions": "crying",
      "outfits": "white dress and black shoes"
    }}
  ],
  "over_three_action_changes": false,
  "over_three_outfit_changes": false,
  "over_three_scene_changes": false
}}

### Output Format:
1. Return only the raw JSON object — do NOT include any commentary, markdown, or explanation.
2. Your output should follow the below structured format (JSON):
```json
{{
  "significant_action_transition": true or false,
  "significant_scene_transition": true or false,
  "significant_outfit_transition": true or false,
  "similar_looking_existence": true or false,
  "justification": {{
    "significant_action_transition": "Your explanation here.",
    "significant_scene_transition": "Your explanation here.",
    "significant_outfit_transition": "Your explanation here.",
    "similar_looking_existence": "If true, describe how the other entity looks and behaves. If false, justify why no such entity appears."
  }},
  "other_entities": [
    {{
      "actions": "Describe actions of other entities in the video.",
      "outfits": "Describe outfits of other entities in the video."
    }}
  ]
}}

Please generate structured metadata for the following target entity information in the video:
Target Entity Information in the Video:
- Action Changes: {action_changes}
- Outfit Changes: {outfit_changes}
- Scene Changes: {scene_changes}
"""

GEMINI_ENTITY_TRACKING_PROMPT_UNSURE = """
  You are given two images: a reference image and a face crop. Your task is to determine whether the two images depict the same person.

  Consider the following factors when making your judgment:
  - Facial features such as eyes, nose, mouth, and face shape
  - Hair style and color (if visible)
  - Accessories (e.g., glasses, earrings) if consistent across both images
  - Contextual clues such as clothing or background, but only if the face is partially visible

  Respond with a structured JSON output:
  {
    "same_identity": true, false, or unsure,
    "justification": "Brief reason for your decision"
  }

  Be conservative in high-confidence decisions and clearly explain any uncertainties in your justification.

  Images:
  - Reference image: [reference.jpg]
  - Face crop: [face_crop.jpg]
  """

GEMINI_ENTITY_TRACKING_PROMPT_WOUNSURE = """
  You are given two images: a reference image and a face crop. Your task is to determine whether the two images depict the same person.

  Consider the following factors when making your judgment:
  - Facial features such as eyes, nose, mouth, and face shape
  - Hair style and color (if visible)
  - Accessories (e.g., glasses, earrings) if consistent across both images
  - Contextual clues such as clothing or background, but only if the face is partially visible

  Respond with a structured JSON output:
  {
    "same_identity": true or false, 
    "justification": "Brief reason for your decision"
  }

  Be conservative in high-confidence decisions and clearly explain any uncertainties in your justification.

  Images:
  - Reference image: [reference.jpg]
  - Face crop: [face_crop.jpg]
  """

GEMINI_ENTITY_TRACKING_PROMPT_WOUNSURE_MULTI = """
You are given a reference image of a person, followed by a set of {n} face crop images. Your task is to determine, for each face crop, whether it depicts the same person as in the reference image.
Do not evaluate the reference image itself. Only compare the face crops to the reference image.

### Important:
- You should analyze all face crop images together, not in isolation.
- Use visual context from the entire set of face crops to help inform your decision for each one.
  For example, if several face crops share similar features or accessories, use those patterns to better judge each one.
- This helps make more consistent and accurate identity decisions.

### Evaluation Criteria:
- Facial features such as eyes, nose, mouth, and face shape
- Hair style and color (if visible)
- Accessories (e.g., glasses, earrings) if consistent across both images
- Contextual clues such as clothing or background, but only if the face is partially visible

### Output Instructions:
- Respond with a JSON array (no explanation, no markdown), where each object corresponds to one face crop, **in the exact order they are given** (excluding the reference).
- For each face crop, include:
  - "same_identity": true or false
  - "justification": A brief explanation for your decision, including any uncertainties

### Output format:
[
  {{"same_identity": true, "justification": "Brief reason for your decision"}},
  {{"same_identity": false, "justification": "Brief reason for your decision"}},
  ...
]

Be conservative in high-confidence matches. Clearly explain uncertainty (e.g., blur, occlusion).

Images:
- First image: reference.jpg (do not evaluate this)
- Following images: [face_crop1.jpg to face_crop{n}.jpg] (evaluate these)
"""

GEMINI_ENTITY_TRACKING_PROMPT_UNSURE_MULTI = """
You are given a reference image of a person, followed by a set of {n} face crop images. Your task is to determine, for each face crop, whether it depicts the same person as in the reference image.
Do not evaluate the reference image itself. Only compare the face crops to the reference image.

### Important:
- You should analyze all face crop images together, not in isolation.
- Use visual context from the entire set of face crops to help inform your decision for each one.
  For example, if several face crops share similar features or accessories, use those patterns to better judge each one.
- This helps make more consistent and accurate identity decisions.

### Evaluation Criteria:
- Facial features such as eyes, nose, mouth, and face shape
- Hair style and color (if visible)
- Accessories (e.g., glasses, earrings) if consistent across both images
- Contextual clues such as clothing or background, but only if the face is partially visible

### Output Instructions:
- Respond with a JSON array (no explanation, no markdown), where each object corresponds to one face crop, **in the exact order they are given** (excluding the reference).
- For each face crop, include:
  - "same_identity": true, false, or unsure
  - "justification": A brief explanation for your decision, including any uncertainties

### Output format:
[
  {{"same_identity": true, "justification": "Brief reason for your decision"}},
  {{"same_identity": false, "justification": "Brief reason for your decision"}},
  ...
]

Be conservative in high-confidence matches. Clearly explain uncertainty (e.g., blur, occlusion).

Images:
- First image: reference.jpg (do not evaluate this)
- Following images: [face_crop1.jpg to face_crop{n}.jpg] (evaluate these)
"""

EVAL_PROMPT = """You are an expert in video-based narrative understanding and entity tracking.

You will be given a video and a question. Your job is to generate an answer to the question based on what you observe in the video.

If the question is multiple choice, you should provide the answer as a single letter (a, b, c, etc.). If the question is binary, you should answer with "yes" or "no". If the question is ordering task, you should answer with a comma-separated string of the correct order (e.g., "b, a, c").  
Generate answer in JSON format with the following fields:

### Output Format:
```json
{{
  "answer": "Your answer here",
  "justification": "A brief explanation of how you arrived at the answer based on the video content"
}}

### Question: {question}
"""

TEMPLATE_CHECK = """You will be given a question and a template. Your job is to check if the question follows the template in terms of semantics.
If the question follows the template, return "yes". If it does not follow the template, return "no".

### Question: {question}
### Template: {template}  
"""

DISTRACTOR_CHECK= """You will be given lists for a target entity and for a comparison pool.
Decide whether the target's {attribute}s are semantically similar to any item in the pool.
If the target's {attribute}s are semantically similar to any item in the pool, return "yes". If they are not semantically similar, return "no".
Please provide a brief justification for your answer as well as the evidence pairs that support your decision.

Generate answer in JSON format with the following fields:

### Output Format:
```json
{{
  "justification": "Your answer here",
  "answer": "yes" or "no,
  "evidence_pairs": [["<target_{attribute}>", "<matched_pool_{attribute}>"], ...]
}}

### Target {attribute} list: {target_list}
### Comparison pool {attribute} list: {distractor_list}
"""

CONSISTENCY_CHECK = """You will be given a first {attribute} and the last {attribute} of the target entity.
Your job is to check if the first {attribute} and the last {attribute} are semantically consistent.
If they are consistent, return "yes". If they are not consistent, return "no".

Generate answer in JSON format with the following fields:
```json
{{
  "justification": "A brief explanation of how you arrived at the answer based on the first and last {attribute}",
  "answer": "yes" or "no"
}}
### First {attribute}: {first_attribute}
### Last {attribute}: {last_attribute}
"""

OPTION_SIMILARITY_CHECK = """You will be given three options (e.g., (a) option 1\n(b) option 2\n(c) option3) for a question.  
Determine whether any two or more of the options are semantically similar in meaning.  
If at least two options are similar, return "yes"; otherwise, return "no".  

Generate answer in JSON format with the following fields:
```json
{{
  "justification": "Brief explanation describing which options are similar (if any) and why",
  "answer": "yes" or "no"
}}

### Options: {options}
"""

OPTION_SIMILARITY_CHECK_LV2 = """You will be given two options.  
Determine whether the two options are semantically similar or whether one option is a higher-level (more general or superset) of the other.
- If the two options are similar (e.g., office vs office with a picture), return "yes".
- If one option is a higher-level that is more general of the other (e.g., indoor room vs office), return "yes".
- If one option is a subset of the other (e.g., black jacket, red long sleeve button-up shirt, and light-colored pants vs red shirt and black jacket), return "yes".
- If two option have some overlapping elements but not exactly similar (e.g., The person entering a coffee shop and standing, wearing a black leather jacket over a dark shirt and jeans vs The person entering a coffee shop and standing, wearing a red coat over a black top), return "no".
- If neither of the above cases apply, return "no".

Generate answer in JSON format with the following fields:
```json
{{
  "justification": "Brief explanation describing which options are similar (if any) and why",
  "answer": "yes" or "no"
}}

### Option1: {option1}
### Option2: {option2}
"""

GRAMMAR_CHECK = """You will be given a template and a question.  
Your task is to determine whether the question:  
1. Is grammatically correct.  
2. Is easy to understand.  

While doing this, ensure the question’s intent remains consistent with the given template.  

Rules:  
- If the question is grammatically correct, set "grammar" to "yes".  
- If the question is not grammatically correct, set "grammar" to a corrected version that preserves its meaning.  
- If the question is easy to understand, set "understandable" to "yes".  
- If the question is not easy to understand, set "understandable" to a corrected version that is easier for the model to understand (without changing the meaning).  

Generate answer in JSON format with the following fields:
```json 
{{
  "justification": "Brief explanation of the grammar correctness and understandability, including any changes made",
  "grammar": "yes" or "corrected question",
  "understandable": "yes" or "corrected question"
}}

### Template: {template}
### Question: {question}
"""