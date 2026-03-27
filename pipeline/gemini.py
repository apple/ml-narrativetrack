#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import os
import json
import re
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Optional

import vertexai
from PIL import Image
from tenacity import retry, stop_after_attempt
from vertexai import generative_models
from vertexai.generative_models import GenerationConfig, GenerativeModel, Part, Tool, FunctionDeclaration, ToolConfig
from vertexai.vision_models import Video


SINGLE_RESPONSE_SCHEMA = FunctionDeclaration(
    name="extract_person_information",
    description="Extract action, outfit, and scene from the video",
    parameters={
        "type": "object",
        "properties": {
            "action": {"type": "string", "description": "What action is being performed?"},
            "outfit": {"type": "string", "description": "Describe the outfit"},
            "scene": {"type": "string", "description": "Describe the background scene"},
        },
        "required": ["action", "outfit", "scene"]
    }
)

ENTIRE_RESPONSE_SCHEMA = FunctionDeclaration(
    name="extract_overall_info",
    description="Extract high-level transitions and justification from the video.",
    parameters={
        "type": "object",
        "properties": {
            "significant_action_transition": {
                "type": "boolean",
                "description": "Whether the target person performs significantly different actions across segments."
            },
            "significant_scene_transition": {
                "type": "boolean",
                "description": "Whether the background or setting changes significantly."
            },
            "significant_outfit_transition": {
                "type": "boolean",
                "description": "Whether the person changes outfits across the segments."
            },
            "similar_looking_existence": {
                "type": "boolean",
                "description": "Whether other people appear who look visually similar to the target person."
            },
            "other_entities": {
                "type": "array",
                "description": "List of other entities in the video, each with their actions and outfits.",
                "items": {
                    "type": "object",
                    "properties": {
                    "actions": {
                        "type": "string",
                        "description": "Describe actions of the entity."
                    },
                    "outfits": {
                        "type": "string",
                        "description": "Describe outfits of the entity."
                    }
                    },
                    "required": ["actions", "outfits"]
                }
            },
            "justification": {
                "type": "object",
                "description": "Textual justification for each boolean decision above.",
                "properties": {
                    "significant_action_transition": {
                        "type": "string",
                        "description": "Justification for action transition decision."
                    },
                    "significant_scene_transition": {
                        "type": "string",
                        "description": "Justification for scene transition decision."
                    },
                    "significant_outfit_transition": {
                        "type": "string",
                        "description": "Justification for outfit transition decision."
                    },
                    "similar_looking_existence": {
                        "type": "string",
                        "description": "Justification for similar looking existence."
                    }
                },
                "required": [
                    "significant_action_transition",
                    "significant_scene_transition",
                    "significant_outfit_transition",
                    "similar_looking_existence"
                ]
            },
            "over_three_action_changes": {
                "type": "boolean",
                "description": "Whether the target person performs significantly different actions across segments."
            },
            "over_three_outfit_changes": {
                "type": "boolean",
                "description": "Whether the target person performs significantly different actions across segments."
            },
            "over_three_scene_changes": {
                "type": "boolean",
                "description": "Whether the target person performs significantly different actions across segments."
            },
        },
        "required": [
            "significant_action_transition",
            "significant_scene_transition",
            "significant_outfit_transition",
            "similar_looking_existence",
            "other_entities",
            "justification",
            "over_three_action_changes",
            "over_three_outfit_changes",
            "over_three_scene_changes",
        ]
    }
)

EVAL_RESPONSE_SCHEMA = FunctionDeclaration(
    name="eval",
    description="Generate a response to a yes/no question based on video content",
    parameters={
        "type": "object",
        "properties": {
            "answer": {"type": "string", "description": "answer to the question"},
            "justification": {"type": "string", "description": "Describe the reason for the answer"},
        },
        "required": ["answer", "justification"]
    }
)

class BaseEngine(ABC):
    def __init__(sel, model_name) -> None:
        pass

    @abstractmethod
    def generate_response(self) -> None:
        pass


class GeminiVideoPipeline(BaseEngine):
    def __init__(self, model_name, schema_type, config: Optional[dict] = None):
        config = config or {}
        # Set up GCP environment using service account, STS proxy does not work for vertex ai API
        # calls.
        project = "PROJECT NAME HERE"

        # Set up authenticator and credentials before vertexai.init()
        vertexai.init(project=project)

        # Initialize parameters for VLM
        self._prompt_template = config.get(
            "prompt_template",
            (
                "You are an assistant for video understanding. Even without a video provided, you can still give a reasonable answer based on common sense and logical reasoning. Please respond with only the letter (Yes or No) of the correct option."
                "Task: \n"
            ),
        )
        self._model_name = model_name
        self._temperature = config.get("temperature", 0)
        self._schema_type = schema_type

        if self._schema_type == "single":
            function_schema = SINGLE_RESPONSE_SCHEMA
        elif self._schema_type == "entire":
            function_schema = ENTIRE_RESPONSE_SCHEMA 
        elif self._schema_type == "eval":
            function_schema = EVAL_RESPONSE_SCHEMA
        tool = Tool(function_declarations=[function_schema])
        self._model = GenerativeModel(model_name=self._model_name, tools=[tool])

        self._config = GenerationConfig(
            temperature=self._temperature,
            top_p=1,
            top_k=32,
        )
        self._safety_config = [
            generative_models.SafetySetting(
                category=generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=generative_models.HarmBlockThreshold.BLOCK_NONE,
            ),
            generative_models.SafetySetting(
                category=generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=generative_models.HarmBlockThreshold.BLOCK_NONE,
            ),
            generative_models.SafetySetting(
                category=generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=generative_models.HarmBlockThreshold.BLOCK_NONE,
            ),
            generative_models.SafetySetting(
                category=generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=generative_models.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]

    @retry(stop=stop_after_attempt(3), retry_error_callback=lambda _: "_ERROR_")
    def generate_response_single(self, prompt_content):
        try:
            response = self._model.generate_content(
                contents=prompt_content,
                generation_config=self._config,
                safety_settings=self._safety_config,
                tool_config=ToolConfig(
                    function_calling_config=ToolConfig.FunctionCallingConfig(
                        mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
                    )
                )
            )   
            for part in response.candidates[0].content.parts:
                if hasattr(part, "function_call"):
                    if part.function_call.name == "extract_person_information":
                        return {
                            "action": part.function_call.args.get("action"),
                            "outfit": part.function_call.args.get("outfit"),
                            "scene": part.function_call.args.get("scene"),
                        }
            return None

        except Exception as e:
            print(f"[Warning] Structured parsing failed: {e}")
            return response.text.strip().lower()

    @retry(stop=stop_after_attempt(3), retry_error_callback=lambda _: "_ERROR_")
    def generate_response_entire(self, prompt_content):
        try:
            response = self._model.generate_content(
                prompt_content,
                generation_config=self._config,
                safety_settings=self._safety_config,
                tool_config=ToolConfig(
                    function_calling_config=ToolConfig.FunctionCallingConfig(
                        mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
                    )
                )
            )
            for part in response.candidates[0].content.parts:
                if hasattr(part, "function_call"):
                    if part.function_call.name == "extract_overall_info":
                        return {
                            "significant_action_transition": part.function_call.args.get("significant_action_transition"),
                            "significant_scene_transition": part.function_call.args.get("significant_scene_transition"),
                            "significant_outfit_transition": part.function_call.args.get("significant_outfit_transition"),
                            "similar_looking_existence": part.function_call.args.get("similar_looking_existence"),
                            "justification": part.function_call.args.get("justification"),
                            "other_entities": part.function_call.args.get("other_entities"),
                            "over_three_action_changes": part.function_call.args.get("over_three_action_changes"),
                            "over_three_outfit_changes": part.function_call.args.get("over_three_outfit_changes"),
                            "over_three_scene_changes": part.function_call.args.get("over_three_scene_changes"),
                        }
            return None

        except Exception as e:
            print(f"[Warning] Structured parsing failed: {e}")
            return response.text.strip().lower()
        
    @retry(stop=stop_after_attempt(3), retry_error_callback=lambda _: "_ERROR_")
    def generate_response_eval(self, prompt_content):
        try:
            response = self._model.generate_content(
                prompt_content,
                generation_config=self._config,
                safety_settings=self._safety_config,
                tool_config=ToolConfig(
                    function_calling_config=ToolConfig.FunctionCallingConfig(
                        mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
                    )
                )
            )
            for part in response.candidates[0].content.parts:
                if hasattr(part, "function_call"):
                    if part.function_call.name == "eval":
                        return {
                            "answer": part.function_call.args.get("answer"),
                            "justification": part.function_call.args.get("justification"),
                        }
            return None

        except Exception as e:
            print(f"[Warning] Structured parsing failed: {e}")
            answer = response.text.strip().lower()
            answer = re.sub(r"```json|```", "", answer).strip()
            try:
                return json.loads(answer)
            except:
                return response.text.strip().lower()

    def generate_response(self, input_text: str, video_path: str, video_id: str, q_type:str=None) -> str:
        video = Video.load_from_file(video_path)
        prompt_content = [
            Part.from_data(video._video_bytes, mime_type="video/mp4"),
            input_text,
        ]
        if self._schema_type == "single":
            return self.generate_response_single(prompt_content)
        elif self._schema_type == "entire":
            return self.generate_response_entire(prompt_content)
        elif self._schema_type == "eval":
            return self.generate_response_eval(prompt_content)
        else:
            raise ValueError(f"Unsupported schema type: {self._schema_type}. Supported types are 'single' and 'entire'.")


ANNOTATION_RESPONSE_SCHEMA = FunctionDeclaration(
    name="entity_tracking_annotation",
    description="Simulate human annotation for entity tracking",
    parameters={
        "type": "object",
        "properties": {
            "annotations": {
                "type": "array",
                "description": "List of structured annotation responses.",
                "items": {
                    "description": "Data for annotation",
                    "type": "object",
                    "properties": {
                        "same_identity": {
                            "type": "boolean",
                            "description": "Whether the reference image and face crop belong to the same person."
                        },
                        "justification": {
                            "type": "string",
                            "description": "Reasoning for the identity decision."
                        }
                    },
                    "required": ["same_identity", "justification"]
                },
            },
        },
        "required": ["annotations"],
    }
)

class GeminiImageIdentityPipeline(BaseEngine):
    def __init__(self, model_name, config: Optional[dict] = None):
        config = config or {}
        project = "PROJECT NAME HERE"

        # Set up authenticator and credentials before vertexai.init()
        vertexai.init(project=project)
        
        # Initialize parameters for VLM
        self._prompt_template = config.get(
            "prompt_template",
            (
                "You are an assistant for video understanding. Even without a video provided, you can still give a reasonable answer based on common sense and logical reasoning. Please respond with only the letter (Yes or No) of the correct option."
                "Task: \n"
            ),
        )
        self._model_name = model_name 
        self._temperature = config.get("temperature", 0)

        self._model = GenerativeModel(model_name=self._model_name, tools=[Tool(function_declarations=[ANNOTATION_RESPONSE_SCHEMA])])
        self._config = GenerationConfig(
            temperature=self._temperature,
            top_p=1,
            top_k=32,
        )
        self._safety_config = [
            generative_models.SafetySetting(
                category=generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=generative_models.HarmBlockThreshold.BLOCK_NONE,
            ),
            generative_models.SafetySetting(
                category=generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=generative_models.HarmBlockThreshold.BLOCK_NONE,
            ),
            generative_models.SafetySetting(
                category=generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=generative_models.HarmBlockThreshold.BLOCK_NONE,
            ),
            generative_models.SafetySetting(
                category=generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=generative_models.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]

    def pil_image_to_part(self, image: Image.Image, mime_type: str = "image/png") -> Part:
        buffer = BytesIO()
        format = "PNG" if mime_type == "image/png" else "JPEG"
        image.save(buffer, format=format)
        buffer.seek(0)
        return Part.from_data(buffer.read(), mime_type=mime_type)

    @retry(stop=stop_after_attempt(10), retry_error_callback=lambda _: "_ERROR_")
    def generate_response(self, input_text: str, reference_image_path, face_crop_paths) -> str:
        # Load both images as PIL.Image
        reference_image = Image.open(reference_image_path)
        reference_part = self.pil_image_to_part(reference_image, mime_type="image/png")
        face_parts = []
        for path in face_crop_paths:
            img = Image.open(path)
            face_parts.append(self.pil_image_to_part(img, mime_type="image/jpeg"))

        # Compose prompt: [ref image, face 1, face 2, ..., prompt text]
        num_target_imgs = len(face_crop_paths)
        prompt_content = [reference_part] + face_parts + [input_text.format(n=num_target_imgs)]
        
        try:
            response = self._model.generate_content(
                prompt_content,
                generation_config=self._config,
                safety_settings=self._safety_config,
                tool_config=ToolConfig(
                    function_calling_config=ToolConfig.FunctionCallingConfig(
                        mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
                    )
                )
            )
            for part in response.candidates[0].content.parts:
                if hasattr(part, "function_call"):
                    if part.function_call.name == "entity_tracking_annotation":
                        args = part.function_call.args['annotations']

                        if isinstance(args, list):
                            if len(args) != num_target_imgs:
                                print(f"Expected {num_target_imgs} annotations, got {len(args)}")
                                return ["_ERROR_"] * num_target_imgs
                            return args 
                            
                        elif isinstance(args, dict):
                            for key, val in args.items():
                                if isinstance(val, list) and all(isinstance(v, dict) for v in val):
                                    return val 
                        raise ValueError("Unexpected format in function_call.args")

        except Exception as e:
            print(f"[ERROR] Gemini generation failed: {e}")
            return ["_ERROR_"] * num_target_imgs


BINARY_QA_SCHEMA = FunctionDeclaration(
    name="question_answer_generation",
    description="QA generation",
    parameters={
        "type": "object",
        "properties": {
            "qa_pairs": {
                "type": "array",
                "description": "List of qa pairs.",
                "items": {
                    "description": "qa pair",
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "question."
                        },
                        "answer": {
                            "type": "string",
                            "description": "answer."
                        },
                    },
                    "required": ["question", "answer"]
                },
            },
        },
        "required": ["qa_pairs"],
    }
)

MCQA_SCHEMA = FunctionDeclaration(
    name="question_answer_generation",
    description="QA generation",
    parameters={
        "type": "object",
        "properties": {
            "qa_pairs": {
                "type": "array",
                "description": "List of qa pairs.",
                "items": {
                    "description": "qa pair",
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "question."
                        },
                        "options": {
                            "type": "object",
                            "description": "multiple choice options",
                            "properties": {
                                "a": {"type": "string", "description": "Option a"},
                                "b": {"type": "string", "description": "Option b"},
                                "c": {"type": "string", "description": "Option c"},
                            },
                            "required": ["a", "b", "c"]
                        },
                        "answer": {
                            "type": "string",
                            "description": "answer."
                        },
                    },
                    "required": ["question", "options", "answer"]
                },
            },
        },
        "required": ["qa_pairs"],
    }
)

DISTRACTOR_SCHEMA = FunctionDeclaration(
    name="distractor_check",
    description="Check semantic similarity and return justification, binary answer, and evidence pairs.",
    parameters={
        "type": "object",
        "properties": {
            "justification": {
                "type": "string",
                "description": "Short justification referencing the matched phrases. Empty if no match."
            },
            "answer": {
                "type": "string",
                "description": "Overall decision: 'yes' if any match exists, otherwise 'no'.",
                "enum": ["yes", "no"]
            },
            "evidence_pairs": {
                "type": "array",
                "description": "List of evidence pairs with matched target and pool values. Empty if no match.",
                "items": {
                    "type": "object",
                    "properties": {
                        "target": {
                            "type": "string",
                            "description": "The target value that was compared."
                        },
                        "pool": {
                            "type": "string",
                            "description": "The matched value from the pool."
                        }
                    },
                    "required": ["target", "pool"]
                }
            }
        },
        "required": ["justification", "answer", "evidence_pairs"]
    }
)

CONSISTENCY_SCHEMA = FunctionDeclaration(
    name="consistency_check",
    description="Check semantic similarity and return justification, binary answer.",
    parameters={
        "type": "object",
        "properties": {
            "justification": {
                "type": "string",
                "description": "Short justification referencing the matched phrases. Empty if no match."
            },
            "answer": {
                "type": "string",
                "description": "Overall decision: 'yes' if any match exists, otherwise 'no'.",
                "enum": ["yes", "no"]
            },
        },
        "required": ["justification", "answer"]
    }
)

OPTION_SCHEMA = FunctionDeclaration(
    name="option_sim_check",
    description="Check semantic similarity and return justification, and binary answer.",
    parameters={
        "type": "object",
        "properties": {
            "justification": {
                "type": "string",
                "description": "Short justification referencing the matched phrases. Empty if no match."
            },
            "answer": {
                "type": "string",
                "description": "Overall decision: 'yes' if any match exists, otherwise 'no'.",
                "enum": ["yes", "no"]
            },
        },
        "required": ["justification", "answer"]
    }
)

GRAMMAR_CHECK_SCHEMA = FunctionDeclaration(
    name="grammar_check",
    description="Check grammar and return justification, and revised question.",
    parameters={
        "type": "object",
        "properties": {
            "justification": {
                "type": "string",
                "description": "Short justification."
            },
            "grammar": {
                "type": "string",
                "description": "Overall decision: 'yes' if it is grammatically correct, otherwise revised question.",
            },
            "understandable": {
                "type": "string",
                "description": "Overall decision: 'yes' if it is easy to understand, otherwise revised question.",
            },
        },
        "required": ["justification", "grammar", "understandable"]
    }
)


class GeminiTextPipeline(BaseEngine):
    def __init__(self, model_name, _type=None, config: Optional[dict] = None):
        config = config or {}
        project = "PROJECT NAME HERE"

        # Set up authenticator and credentials before vertexai.init()
        vertexai.init(project=project)
        
        # Initialize parameters for VLM
        self._prompt_template = config.get(
            "prompt_template",
            (
                "You are an assistant for video understanding. Even without a video provided, you can still give a reasonable answer based on common sense and logical reasoning. Please respond with only the letter (Yes or No) of the correct option."
                "Task: \n"
            ),
        )
        self._model_name = model_name 
        self._temperature = config.get("temperature", 0)
        
        if _type is None:
            self._model = GenerativeModel(model_name=self._model_name)
        elif _type == "distractor":
            schema = DISTRACTOR_SCHEMA
            self._model = GenerativeModel(model_name=self._model_name, tools=[Tool(function_declarations=[schema])])
        elif _type == "consistency":
            schema = CONSISTENCY_SCHEMA
            self._model = GenerativeModel(model_name=self._model_name, tools=[Tool(function_declarations=[schema])])
        elif _type == "option_similarity":
            schema = OPTION_SCHEMA
            self._model = GenerativeModel(model_name=self._model_name, tools=[Tool(function_declarations=[schema])])
        elif _type == "grammar_check":
            schema = GRAMMAR_CHECK_SCHEMA
            self._model = GenerativeModel(model_name=self._model_name, tools=[Tool(function_declarations=[schema])])
        else:
            schema = BINARY_QA_SCHEMA if _type=="binary" else MCQA_SCHEMA
            self._model = GenerativeModel(model_name=self._model_name, tools=[Tool(function_declarations=[schema])])
        
        self._config = GenerationConfig(
            temperature=self._temperature,
            top_p=1,
            top_k=32,
        )
        self._safety_config = [
            generative_models.SafetySetting(
                category=generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=generative_models.HarmBlockThreshold.BLOCK_NONE,
            ),
            generative_models.SafetySetting(
                category=generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=generative_models.HarmBlockThreshold.BLOCK_NONE,
            ),
            generative_models.SafetySetting(
                category=generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=generative_models.HarmBlockThreshold.BLOCK_NONE,
            ),
            generative_models.SafetySetting(
                category=generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=generative_models.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]

    @retry(stop=stop_after_attempt(10), retry_error_callback=lambda _: "_ERROR_")
    def generate_response(self, input_text) -> str:
        try:
            response = self._model.generate_content(
                input_text,
                generation_config=self._config,
                safety_settings=self._safety_config,
                tool_config=ToolConfig(
                    function_calling_config=ToolConfig.FunctionCallingConfig(
                        mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
                    )
                )
            )
            for part in response.candidates[0].content.parts:
                if hasattr(part, "function_call"):
                    if part.function_call.name == "question_answer_generation":
                        args = part.function_call.args['qa_pairs']
                        if isinstance(args, list):
                            return args 
                            
                        elif isinstance(args, dict):
                            for key, val in args.items():
                                if isinstance(val, list) and all(isinstance(v, dict) for v in val):
                                    return val 
                        raise ValueError("Unexpected format in function_call.args")

        except Exception as e:
            print(f"[ERROR] QA generation failed: {e}")
            return "_ERROR_"
    
    @retry(stop=stop_after_attempt(2), retry_error_callback=lambda _: "_ERROR_")
    def generate_response_base(self, input_text) -> str:
        try:
            response = self._model.generate_content(
                input_text,
                generation_config=self._config,
                safety_settings=self._safety_config
            )
            return response.text.strip().lower()
        except Exception as e:
            print(f"[ERROR] QA generation failed: {e}")
            return "_ERROR_"

    @retry(stop=stop_after_attempt(10), retry_error_callback=lambda _: "_ERROR_")
    def generate_response_distractor(self, input_text) -> str:
        try:
            response = self._model.generate_content(
                input_text,
                generation_config=self._config,
                safety_settings=self._safety_config,
                tool_config=ToolConfig(
                    function_calling_config=ToolConfig.FunctionCallingConfig(
                        mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
                    )
                )
            )

            for part in response.candidates[0].content.parts:
                if hasattr(part, "function_call"):
                    if part.function_call.name == "distractor_check":
                        return {
                            "justification": part.function_call.args.get("justification"),
                            "answer": part.function_call.args.get("answer"),
                            "evidence_pairs": part.function_call.args.get("evidence_pairs"),
                        }
            return None

        except Exception as e:
            print(f"[ERROR] QA generation failed: {e}")
            return "_ERROR_"
    
    @retry(stop=stop_after_attempt(10), retry_error_callback=lambda _: "_ERROR_")
    def generate_response_consistency(self, input_text) -> str:
        try:
            response = self._model.generate_content(
                input_text,
                generation_config=self._config,
                safety_settings=self._safety_config,
                tool_config=ToolConfig(
                    function_calling_config=ToolConfig.FunctionCallingConfig(
                        mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
                    )
                )
            )

            for part in response.candidates[0].content.parts:
                if hasattr(part, "function_call"):
                    if part.function_call.name == "consistency_check":
                        return {
                            "justification": part.function_call.args.get("justification"),
                            "answer": part.function_call.args.get("answer"),
                        }
            return None

        except Exception as e:
            print(f"[ERROR] QA generation failed: {e}")
            return "_ERROR_"

    @retry(stop=stop_after_attempt(10), retry_error_callback=lambda _: "_ERROR_")
    def generate_response_option_similarity(self, input_text) -> str:
        try:
            response = self._model.generate_content(
                input_text,
                generation_config=self._config,
                safety_settings=self._safety_config,
                tool_config=ToolConfig(
                    function_calling_config=ToolConfig.FunctionCallingConfig(
                        mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
                    )
                )
            )

            for part in response.candidates[0].content.parts:
                if hasattr(part, "function_call"):
                    if part.function_call.name == "option_sim_check":
                        return {
                            "justification": part.function_call.args.get("justification"),
                            "answer": part.function_call.args.get("answer"),
                        }
            return None

        except Exception as e:
            print(f"[ERROR] QA generation failed: {e}")
            return "_ERROR_"

    @retry(stop=stop_after_attempt(10), retry_error_callback=lambda _: "_ERROR_")
    def generate_response_grammar_check(self, input_text) -> str:
        try:
            response = self._model.generate_content(
                input_text,
                generation_config=self._config,
                safety_settings=self._safety_config,
                tool_config=ToolConfig(
                    function_calling_config=ToolConfig.FunctionCallingConfig(
                        mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
                    )
                )
            )

            for part in response.candidates[0].content.parts:
                if hasattr(part, "function_call"):
                    if part.function_call.name == "grammar_check":
                        return {
                            "justification": part.function_call.args.get("justification"),
                            "grammar": part.function_call.args.get("grammar"),
                            "understandable": part.function_call.args.get("understandable"),
                        }
            return None

        except Exception as e:
            print(f"[ERROR] QA generation failed: {e}")
            return "_ERROR_"



class GeminiTextEvalPipeline(BaseEngine):
    def __init__(self, model_name, _type=None, config: Optional[dict] = None):
        config = config or {}
        project = "PROJECT NAME HERE"

        # Set up authenticator and credentials before vertexai.init()
        vertexai.init(project=project)

        self._prompt_template = config.get(
            "prompt_template",
            (
                "You are an assistant for video understanding. Even without a video provided, you can still give a reasonable answer based on common sense and logical reasoning. Please respond with only the letter (Yes or No) of the correct option."
                "Task: \n"
            ),
        )
        self._model_name = model_name 
        self._temperature = config.get("temperature", 0)
        
        if _type is None:
            self._model = GenerativeModel(model_name=self._model_name)
        elif _type == "eval":
            function_schema = EVAL_RESPONSE_SCHEMA
            tool = Tool(function_declarations=[function_schema])
            self._model = GenerativeModel(model_name=self._model_name, tools=[tool])
        
        self._config = GenerationConfig(
            # max_output_tokens=self._max_output_tokens,
            temperature=self._temperature,
            top_p=1,
            top_k=32,
        )
        self._safety_config = [
            generative_models.SafetySetting(
                category=generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=generative_models.HarmBlockThreshold.BLOCK_NONE,
            ),
            generative_models.SafetySetting(
                category=generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=generative_models.HarmBlockThreshold.BLOCK_NONE,
            ),
            generative_models.SafetySetting(
                category=generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=generative_models.HarmBlockThreshold.BLOCK_NONE,
            ),
            generative_models.SafetySetting(
                category=generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=generative_models.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]

    @retry(stop=stop_after_attempt(3), retry_error_callback=lambda _: "_ERROR_")
    def generate_response(self, input_text: str, video_path: str, video_id: str, q_type:str=None) -> str:
        try:
            response = self._model.generate_content(
                input_text,
                generation_config=self._config,
                safety_settings=self._safety_config,
                tool_config=ToolConfig(
                    function_calling_config=ToolConfig.FunctionCallingConfig(
                        mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
                    )
                )
            )
            for part in response.candidates[0].content.parts:
                if hasattr(part, "function_call"):
                    if part.function_call.name == "eval":
                        return {
                            "answer": part.function_call.args.get("answer"),
                            "justification": part.function_call.args.get("justification"),
                        }
            return None

        except Exception as e:
            print(f"[ERROR] QA generation failed: {e}")
            return "_ERROR_"
