#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

BINARY_ENTITY_EXISTENCE = {
    "appear": [
        "Is the person {action} in {scene} with {outfit} only seen at the end?",
        "Does the person {action} and wearing {outfit} in {scene} first appear at the end?",
        "Is the person {action} and wearing {outfit} in {scene} not seen earlier but appearing at the end?",
        "Does the person {action} and wearing {outfit} in {scene} appear only at the end?",
        "At the end, is the person {action} and in {outfit} in {scene} seen for the first time?"
    ],
    "reappear": [
        # ---------start to later----------
        "Does the person {action} and wearing {outfit} in {scene} at the beginning disappear during the video and reappear at the end?",
        "Is the person {action} and wearing {outfit} in {scene} from the beginning gone for a while and then present again at the end?",
        "Does the person {action} and in {outfit} in {scene} at the beginning disappear and later show up at the end?",
        "Does the person {action} and wearing {outfit} in {scene} at the beginning appear again at the end?",
        "After appearing {action} and wearing {outfit} in {scene} at the beginning, does the person reappear at the end?",
        # --------later to start------------
        "Does the person {action} and wearing {outfit} in {scene} at the end also appear at the beginning, then disappear for a while?",
        "Is the person {action} and wearing {outfit} in {scene} from the end also appear at the beginning, then gone for a while?",
        "Does the person {action} and in {outfit} in {scene} at the end earlier show up at the beginning?",
        "Does the person {action} and wearing {outfit} in {scene} at the end appear again at the beginning?",
        "After appearing {action} and wearing {outfit} in {scene} at the end, does the person reappear at the beginning?",
    ],
    "disappear": [
        "Does the person with {action} and {outfit} in {scene} appear at the beginning and then remain unseen afterward?",
        "Is the person {action} in {scene} with {outfit} seen only at the beginning, then gone?",
        "Does the person {action} in {scene} with {outfit} disappear after the beginning and not come back?",
        "Is the person {action} and wearing {outfit} in {scene} only seen at the beginning and never again?",
        "Does the person with {action} and {outfit} in {scene} appear at the beginning, then leave and never come back?",
    ]
}

BINARY_ACTION_CHANGES = {
    "start_to_later": [
        "Is the person {action1} in {scene} with {outfit} at the beginning later performing {action2}?",
        "Is the person {action1} and wearing {outfit} in {scene} at the beginning later performing {action2}?",
        "Is the person who is {action1} and in {outfit} in {scene} at the beginning later performing {action2}?",
        "At the beginning, the person {action1} in {scene} with {outfit} is visible — do they perform {action2} later?",
        "Does the person with {action1} and {outfit} in {scene} at the beginning later performing {action2}?",
    ],
    "later_to_start": [
        "Is the person {action1} in {scene} with {outfit} at the end earlier performing {action2}?",
        "Is the person {action1} and wearing {outfit} in {scene} at the end earlier performing {action2}?",
        "Is the person who is {action1} and in {outfit} in {scene} at the end earlier performing {action2}?",
        "At the end, the person {action1} in {scene} with {outfit} is visible — do they perform {action2} earlier?",
        "Does the person with {action1} and {outfit} in {scene} at the end earlier performing {action2}?",
    ]
}

BINARY_SCENE_CHANGES = {
    "start_to_later": [
        "Does the person {action} and wearing {outfit} in {scene1} at the beginning appear later in {scene2}?",
        "Is the person {action} and wearing {outfit} in {scene1} at the beginning seen in {scene2} later?",
        "Is the person {action} and in {scene1} in {outfit} at the beginning shown in {scene2} later?",
        "After {action} and wearing {outfit} in {scene1} at the beginning, does the person show up in {scene2} later?",
        "Is the person {action} with {outfit} in {scene1} at the beginning present in {scene2} later?",
    ],
    "later_to_start": [
        "Does the person {action} and wearing {outfit} in {scene1} at the end appear earlier in {scene2}?",
        "Is the person {action} and wearing {outfit} in {scene1} at the end seen in {scene2} earlier?",
        "Is the person {action} and in {scene1} in {outfit} at the end shown in {scene2} earlier?",
        "Before {action} and wearing {outfit} in {scene1} at the end, does the person show up in {scene2} earlier?",
        "Is the person {action} with {outfit} in {scene1} at the end present in {scene2} earlier?",
    ]
}

BINARY_OUTFIT_CHANGES = {
    "start_to_later": [
        "Is the person {action} and wearing {outfit1} in {scene} at the beginning shown later wearing {outfit2}?",
        "At the beginning, the person is {action} in {scene} with {outfit1} — do they wear {outfit2} later?",
        "Is the person {action} and in {outfit1} in {scene} at the beginning seen later in {outfit2}?",
        "After being in {scene} {action} and wearing {outfit1} at the beginning, is the person later seen wearing {outfit2}?",
        "Does the person {action} and wearing {outfit1} in {scene} at the beginning later seen in {outfit2}?"
    ],
    "later_to_start": [
        "Is the person {action} and wearing {outfit1} in {scene} at the end shown earlier wearing {outfit2}?",
        "At the end, the person is {action} in {scene} with {outfit1} — do they wear {outfit2} earlier?",
        "Is the person {action} and in {outfit1} in {scene} at the end seen earlier in {outfit2}?",
        "Before being in {scene} {action} and wearing {outfit1} at the end, is the person earlier seen wearing {outfit2}?",
        "Does the person {action} and wearing {outfit1} in {scene} at the end earlier seen in {outfit2}?"
    ]
}

BINARY_ENTITY_AMBIGUITY = {
    "start_to_later": [
        "Is the person {action1} and wearing {outfit1} in {scene1} at the beginning the same person seen later with {action2}, {outfit2}, and {scene2}?",
        "Does the person with {action1} and in {outfit1} in {scene1} at the beginning match the same person seen later with {action2}, {outfit2}, and {scene2}?",
        "At the beginning, the person is {action1} in {scene1} with {outfit1} — is the same person seen later with {action2}, {outfit2}, and {scene2}?",
        "After appearing {action1} and wearing {outfit1} in {scene1} at the beginning, is it the same person shown later with {action2}, {outfit2}, and {scene2}?",
        "Is the person {action1} in {scene1} with {outfit1} at the beginning identical to the person later {action2} in {scene2} with {outfit2}?",
    ],
    "start_to_later_wo_scene": [
        "Is the person {action1} and wearing {outfit1} in {scene1} at the beginning the same person seen later with {action2}, {outfit2}?",
        "Does the person with {action1} and in {outfit1} in {scene1} at the beginning match the same person seen later with {action2}, {outfit2}?",
        "At the beginning, the person is {action1} in {scene1} with {outfit1} — is the same person seen later with {action2}, {outfit2}?",
        "After appearing {action1} and wearing {outfit1} in {scene1} at the beginning, is it the same person shown later with {action2}, {outfit2}?",
        "Is the person {action1} in {scene1} with {outfit1} at the beginning identical to the person later {action2} with {outfit2}?",
    ],
    "later_to_start": [
        "Is the person {action1} and wearing {outfit1} in {scene1} at the end the same person seen earlier with {action2}, {outfit2}, and {scene2}?",
        "Does the person with {action1} and in {outfit1} in {scene1} at the end match the same person seen earlier with {action2}, {outfit2}, and {scene2}?",
        "At the end, the person is {action1} in {scene1} with {outfit1} — is the same person seen earlier with {action2}, {outfit2}, and {scene2}?",
        "Before appearing {action1} and wearing {outfit1} in {scene1} at the end, is it the same person shown earlier with {action2}, {outfit2}, and {scene2}?",
        "Is the person {action1} in {scene1} with {outfit1} at the end identical to the person earlier {action2} in {scene2} with {outfit2}?",
    ],
    "later_to_start_wo_scene": [
        "Is the person {action1} and wearing {outfit1} in {scene1} at the end the same person seen earlier with {action2}, {outfit2}?",
        "Does the person with {action1} and in {outfit1} in {scene1} at the end match the same person seen earlier with {action2}, {outfit2}?",
        "At the end, the person is {action1} in {scene1} with {outfit1} — is the same person seen earlier with {action2}, {outfit2}?",
        "Before appearing {action1} and wearing {outfit1} in {scene1} at the end, is it the same person shown earlier with {action2}, {outfit2}?",
        "Is the person {action1} in {scene1} with {outfit1} at the end identical to the person earlier {action2} with {outfit2}?",
    ]
}

MC_ENTITY_EXISTENCE = {
    "appear": [ # answers: (a)
        "Which best describes the person {action} and wearing {outfit} in {scene} at the end?\n(a) Appear only at the end\n(b) Appear at the start, missing for a while, then back\n(c) Appear at the start, missing until the end",
        "Which best describes the person in {scene} {action} and wearing {outfit} at the end?\n(a) Appear only at the end\n(b) Appear at the start, missing for a while, then back\n(c) Appear at the start, missing until the end",
        "Which best describes the person {action} in {scene} in {outfit} at the end?\n(a) Appears at the end\n(b) Appears at start, disappears, then back(c) Appears at start, disappears, and never back",
        "When is the person appearing at the end with {action} and wearing {outfit} in {scene} seen?\n(a) Only at the end\n(b) At start and end with absence in between\n(c) At start only, then disappears",
        "When is the person appearing at the end with {action} in {scene} wearing {outfit} seen?\n(a) Only at the end\n(b) At start and end with absence in between\n(c) At start only, then disappears",
    ],
    "reappear": [ # answers: (b) 
        # ---------start to later----------
        "Which best describes the person {action} and wearing {outfit} in {scene} at the beginning?\n(a) Appear only at the end\n(b) Appear at the start, missing for a while, then back\n(c) Appear at the start, missing until the end",
        "Which best describes the person in {scene} {action} and wearing {outfit} at the beginning?\n(a) Appear only at the end\n(b) Appear at the start, missing for a while, then back\n(c) Appear at the start, missing until the end",
        "Which best describes the person {action} in {scene} in {outfit} at the beginning?\n(a) Appears at the end\n(b) Appears at start, disappears, then back(c) Appears at start, disappears, and never back",
        "When is the person appearing at the beginning with {action} and wearing {outfit} in {scene} seen?\n(a) Only at the end\n(b) At start and end with absence in between\n(c) At start only, then disappears",
        "When is the person appearing at the beginning with {action} in {scene} wearing {outfit} seen?\n(a) Only at the end\n(b) At start and end with absence in between\n(c) At start only, then disappears",
        # ---------later to start----------
        "Which best describes the person {action} and wearing {outfit} in {scene} at the beginning?\n(a) Appear only at the end\n(b) Appear at the start, missing for a while, then back\n(c) Appear at the start, missing until the end",
        "Which best describes the person in {scene} {action} and wearing {outfit} at the beginning?\n(a) Appear only at the end\n(b) Appear at the start, missing for a while, then back\n(c) Appear at the start, missing until the end",
        "Which best describes the person {action} in {scene} in {outfit} at the beginning?\n(a) Appears at the end\n(b) Appears at start, disappears, then back(c) Appears at start, disappears, and never back",
        "When is the person appearing at the beginning with {action} and wearing {outfit} in {scene} seen?\n(a) Only at the end\n(b) At start and end with absence in between\n(c) At start only, then disappears",
        "When is the person appearing at the beginning with {action} in {scene} wearing {outfit} seen?\n(a) Only at the end\n(b) At start and end with absence in between\n(c) At start only, then disappears",
    ],
    "disappear": [ # answers (c)
        "Which best describes the person {action} and wearing {outfit} in {scene} at the beginning?\n(a) Appear only at the end\n(b) Appear at the start, missing for a while, then back\n(c) Appear at the start, missing until the end",
        "Which best describes the person in {scene} {action} and wearing {outfit} at the beginning?\n(a) Appear only at the end\n(b) Appear at the start, missing for a while, then back\n(c) Appear at the start, missing until the end",
        "Which best describes the person {action} in {scene} in {outfit} at the beginning?\n(a) Appears at the end\n(b) Appears at start, disappears, then back(c) Appears at start, disappears, and never back",
        "When is the person appearing at the beginning with {action} and wearing {outfit} in {scene} seen?\n(a) Only at the end\n(b) At start and end with absence in between\n(c) At start only, then disappears",
        "When is the person appearing at the beginning with {action} in {scene} wearing {outfit} seen?\n(a) Only at the end\n(b) At start and end with absence in between\n(c) At start only, then disappears",
    ]
}

MC_ACTION_CHNAGES = {
    "start_to_later": [
        "What action is the person {action} in {scene} with {outfit} at the beginning later performing?\n(a) {action1}\n(b) {action2}\n(c) {action3}",
        "What action is the person {action} and wearing {outfit} in {scene} at the beginning later performing?\n(a) {action1}\n(b) {action2}\n(c) {action3}",
        "What action is performed later by the person who is {action} and in {outfit} in {scene} at the beginning?\n(a) {action1}\n(b) {action2}\n(c) {action3}",
        "At the beginning, the person {action} in {scene} with {outfit} is visible — what action do they perform later?\n(a) {action1}\n(b) {action2}\n(c) {action3}",
        "What action does the person with {action} and {outfit} in {scene} at the beginning later perform?\n(a) {action1}\n(b) {action2}\n(c) {action3}",
    ],
    "later_to_start": [
        "What action is the person {action} in {scene} with {outfit} at the end earlier performing?\n(a) {action1}\n(b) {action2}\n(c) {action3}",
        "What action is the person {action} and wearing {outfit} in {scene} at the end earlier performing?\n(a) {action1}\n(b) {action2}\n(c) {action3}",
        "What action is performed earlier by the person who is {action} and in {outfit} in {scene} at the end?\n(a) {action1}\n(b) {action2}\n(c) {action3}",
        "At the end, the person {action} in {scene} with {outfit} is visible — what action do they perform earlier?\n(a) {action1}\n(b) {action2}\n(c) {action3}",
        "What action does the person with {action} and {outfit} in {scene} at the end earlier perform?\n(a) {action1}\n(b) {action2}\n(c) {action3}",
    ]
}

MC_SCENE_CHNAGES = {
    "start_to_later": [
        "In which scene does the person {action} and wearing {outfit} in {scene} at the beginning appear later?\n(a) {scene1}\n(b) {scene2}\n (c) {scene3}",
        "In which scene is the person {action} and wearing {outfit} in {scene} at the beginning seen later?\n(a) {scene1}\n(b) {scene2}\n (c) {scene3}",
        "In which scene is the person {action} and in {scene} in {outfit} at the beginning shown later?\n(a) {scene1}\n(b) {scene2}\n (c) {scene3}",
        "After {action} and wearing {outfit} in {scene} at the beginning, in which scene does the person show up later?\n(a) {scene1}\n(b) {scene2}\n (c) {scene3}",
        "In which scene is the person {action} with {outfit} in {scene} at the beginning present later?\n(a) {scene1}\n(b) {scene2}\n (c) {scene3}",
 ],
    "later_to_start": [
        "In which scene does the person {action} and wearing {outfit} in {scene} at the end appear earlier?\n(a) {scene1}\n(b) {scene2}\n (c) {scene3}",
        "In which scene is the person {action} and wearing {outfit} in {scene} at the end seen earlier?\n(a) {scene1}\n(b) {scene2}\n (c) {scene3}",
        "In which scene is the person {action} and in {scene} in {outfit} at the end shown earlier?\n(a) {scene1}\n(b) {scene2}\n (c) {scene3}",
        "After {action} and wearing {outfit} in {scene} at the end, in which scene does the person show up earlier?\n(a) {scene1}\n(b) {scene2}\n (c) {scene3}",
        "In which scene is the person {action} with {outfit} in {scene} at the end present earlier?\n(a) {scene1}\n(b) {scene2}\n (c) {scene3}",
    ]
}

MC_OUTFIT_CHNAGES = {
    "start_to_later": [
        "What outfit is worn by the the person {action} and wearing {outfit} in {scene} at the beginning shown later?\n(a) {outfit1}\n(b) {outfit2}\n (c) {outfit3}",
        "At the beginning, the person is {action} in {scene} with {outfit} — what outfit does the person wear later?\n(a) {outfit1}\n(b) {outfit2}\n (c) {outfit3}",
        "What outfit is the person {action} and in {outfit} in {scene} at the beginning wearing later?\n(a) {outfit1}\n(b) {outfit2}\n (c) {outfit3}",
        "After being in {scene} {action} and wearing {outfit} at the beginning, what outfit is the person later seen wearing?\n(a) {outfit1}\n(b) {outfit2}\n (c) {outfit3}",
        "What outfit is later worn by the person {action} and wearing {outfit} in {scene} at the beginning?\n(a) {outfit1}\n(b) {outfit2}\n (c) {outfit3}",
    ],
    "later_to_start": [
        "What outfit is worn by the the person {action} and wearing {outfit} in {scene} at the end shown earlier?\n(a) {outfit1}\n(b) {outfit2}\n (c) {outfit3}",
        "At the end, the person is {action} in {scene} with {outfit} — what outfit does the person wear earlier?\n(a) {outfit1}\n(b) {outfit2}\n (c) {outfit3}",
        "What outfit is the person {action} and in {outfit} in {scene} at the end wearing earlier?\n(a) {outfit1}\n(b) {outfit2}\n (c) {outfit3}",
        "After being in {scene} {action} and wearing {outfit} at the end, what outfit is the person earlier seen wearing?\n(a) {outfit1}\n(b) {outfit2}\n (c) {outfit3}",
        "What outfit is earlier worn by the person {action} and wearing {outfit} in {scene} at the end?\n(a) {outfit1}\n(b) {outfit2}\n (c) {outfit3}",
    ]
}

MC_ENTITY_AMBIGUITY = {
    "start_to_later": [
        "Later in the video, which person is most likely the same one seen at the beginning {action} and wearing {outfit} in the {scene}?\n(a) {option1}\n(b) {option2}\n(c) {option3}",
        "Later in the video, who is the same person from the beginning in {scene} {action} and wearing {outfit}?\n(a) {option1}\n(b) {option2}\n(c) {option3}",
        "Which person later seen matches the one from the beginning {action} and wearing {outfit} in {scene}?\n(a) {option1}\n(b) {option2}\n(c) {option3}",
        "Later in the video, who is the same person that was at the beginning wearing {outfit} in {scene} {action}?\n(a) {option1}\n(b) {option2}\n(c) {option3}",
        "Which person shown up later matches the one seen at the beginning {action} and wearing {outfit} in the {scene}?\n(a) {option1}\n(b) {option2}\n(c) {option3}",
    ],
    "later_to_start": [ 
        "Earlier in the video, which person is most likely the same one seen at the end {action} and wearing {outfit} in the {scene}?\n(a) {option1}\n(b) {option2}\n(c) {option3}",
        "Earlier in the video, who is the same person from the end in {scene} {action} and wearing {outfit}?\n(a) {option1}\n(b) {option2}\n(c) {option3}",
        "Which person earlier seen matches the one from the end {action} and wearing {outfit} in {scene}?\n(a) {option1}\n(b) {option2}\n(c) {option3}",
        "Earlier in the video, who is the same person that was at the end wearing {outfit} in {scene} {action}?\n(a) {option1}\n(b) {option2}\n(c) {option3}",
        "Which person shown up earlier matches the one seen at the end {action} and wearing {outfit} in the {scene}?\n(a) {option1}\n(b) {option2}\n(c) {option3}",
    ]
}

ACTION_ORDERING = {
    "agnostic": [
        "What is the chronological order of the following actions performed by the person who was seen in the video {action} and wearing {outfit} in {scene}?\n(a) {action1}\n(b) {action2}\n(c) {action3}",
        "Arrange the following actions in the order done by the person seen {action} and wearing {outfit} in {scene} in the video.\n(a) {action1}\n(b) {action2}\n(c) {action3}",
        "In what order did the person {action} and wearing {outfit} in {scene} in the video, perform these actions?\n(a) {action1}\n(b) {action2}\n(c) {action3}",
        "Put the following actions in the order made by the person {action} and wearing {outfit} in {scene} in the video.\n(a) {action1}\n(b) {action2}\n(c) {action3}",
        "Identify the order of actions made by the person {action} and wearing {outfit} in {scene} in the video.\n(a) {action1}\n(b) {action2}\n(c) {action3}"
    ],
    "start_to_later": [
        "What is the chronological order of the following actions performed by the person who was seen at the beginning {action} and wearing {outfit} in {scene}?\n(a) {action1}\n(b) {action2}\n(c) {action3}",
        "Arrange the following actions in the order done by the person seen {action} and wearing {outfit} in {scene} at the beginning.\n(a) {action1}\n(b) {action2}\n(c) {action3}",
        "In what order did the person {action} and wearing {outfit} in {scene} at the beginning, perform these actions?\n(a) {action1}\n(b) {action2}\n(c) {action3}",
        "Put the following actions in the order made by the person {action} and wearing {outfit} in {scene} at the beginning.\n(a) {action1}\n(b) {action2}\n(c) {action3}",
        "Identify the order of actions made by the person {action} and wearing {outfit} in {scene} at the beginning.\n(a) {action1}\n(b) {action2}\n(c) {action3}"
    ],
    "later_to_start": [
        "What is the chronological order of the following actions performed by the person who was seen at the end {action} and wearing {outfit} in {scene}?\n(a) {action1}\n(b) {action2}\n(c) {action3}",
        "Arrange the following actions in the order done by the person seen {action} and wearing {outfit} in {scene} at the end.\n(a) {action1}\n(b) {action2}\n(c) {action3}",
        "In what order did the person {action} and wearing {outfit} in {scene} at the end, perform these actions?\n(a) {action1}\n(b) {action2}\n(c) {action3}",
        "Put the following actions in the order made by the person {action} and wearing {outfit} in {scene} at the end.\n(a) {action1}\n(b) {action2}\n(c) {action3}",
        "Identify the order of actions made by the person {action} and wearing {outfit} in {scene} at the end.\n(a) {action1}\n(b) {action2}\n(c) {action3}"
    ]
}


SCENE_ORDERING = {
    "agnostic": [
        "What is the chronological order of the following scenes involving the person who was seen in the video {action} and wearing {outfit} in {scene}?\n(a) {scene1}\n(b) {scene2}\n(c) {scene3}",
        "Arrange the following scenes in the order the person seen {action} and wearing {outfit} in {scene} in the video.\n(a) {scene1}\n(b) {scene2}\n(c) {scene3}",
        "In what order did the person {action} and wearing {outfit} in {scene} in the video, move through these scenes?\n(a) {scene1}\n(b) {scene2}\n(c) {scene3}",
        "Put the following scenes in the order the person seen {action} and wearing {outfit} in {scene} in the video.\n(a) {scene1}\n(b) {scene2}\n(c) {scene3}",
        "Identify the order of scenes the person shown up with {action} and wearing {outfit} in {scene} in the video.\n(a) {scene1}\n(b) {scene2}\n(c) {scene3}",
    ],
    "start_to_later": [
        "What is the chronological order of the following scenes involving the person who was seen at the beginning {action} and wearing {outfit} in {scene}?\n(a) {scene1}\n(b) {scene2}\n(c) {scene3}",
        "Arrange the following scenes in the order the person seen {action} and wearing {outfit} in {scene} at the beginning.\n(a) {scene1}\n(b) {scene2}\n(c) {scene3}",
        "In what order did the person {action} and wearing {outfit} in {scene} at the beginning, move through these scenes?\n(a) {scene1}\n(b) {scene2}\n(c) {scene3}",
        "Put the following scenes in the order the person seen {action} and wearing {outfit} in {scene} at the beginning.\n(a) {scene1}\n(b) {scene2}\n(c) {scene3}",
        "Identify the order of scenes the person shown up with {action} and wearing {outfit} in {scene} at the beginning.\n(a) {scene1}\n(b) {scene2}\n(c) {scene3}",
    ],
    "later_to_start": [
        "What is the chronological order of the following scenes involving the person who was seen at the end {action} and wearing {outfit} in {scene}?\n(a) {scene1}\n(b) {scene2}\n(c) {scene3}",
        "Arrange the following scenes in the order the person seen {action} and wearing {outfit} in {scene} at the end.\n(a) {scene1}\n(b) {scene2}\n(c) {scene3}",
        "In what order did the person {action} and wearing {outfit} in {scene} at the end, move through these scenes?\n(a) {scene1}\n(b) {scene2}\n(c) {scene3}",
        "Put the following scenes in the order the person seen {action} and wearing {outfit} in {scene} at the end.\n(a) {scene1}\n(b) {scene2}\n(c) {scene3}",
        "Identify the order of scenes the person shown up with {action} and wearing {outfit} in {scene} at the end.\n(a) {scene1}\n(b) {scene2}\n(c) {scene3}",
    ]
}

OUTFIT_ORDERING = {
    "agnostic": [
        "What is the chronological order of the following outfits worn by the person who was seen in the video {action} and wearing {outfit} in {scene}?\n(a) {outfit1}\n(b) {outfit2}\n(c) {outfit3}",
        "Arrange the following outfits in the order worn by the person {action} and wearing {outfit} in {scene} in the video.\n(a) {outfit1}\n(b) {outfit2}\n(c) {outfit3}",
        "In what order did the person {action} and wearing {outfit} in {scene} in the video, worn through the video?\n(a) {outfit1}\n(b) {outfit2}\n(c) {outfit3}",
        "Put the following outfits in the order worn by the person {action} and wearing {outfit} in {scene} in the video.\n(a) {outfit1}\n(b) {outfit2}\n(c) {outfit3}",
        "Identify the order of outfits worn by the person shown up with {action} and wearing {outfit} in {scene} in the video.\n(a) {outfit1}\n(b) {outfit2}\n(c) {outfit3}",
    ],
    "start_to_later": [
        "What is the chronological order of the following outfits worn by the person who was seen at the beginning {action} and wearing {outfit} in {scene}?\n(a) {outfit1}\n(b) {outfit2}\n(c) {outfit3}",
        "Arrange the following outfits in the order worn by the person {action} and wearing {outfit} in {scene} at the beginning.\n(a) {outfit1}\n(b) {outfit2}\n(c) {outfit3}",
        "In what order did the person {action} and wearing {outfit} in {scene} at the beginning, worn through the video?\n(a) {outfit1}\n(b) {outfit2}\n(c) {outfit3}",
        "Put the following outfits in the order worn by the person {action} and wearing {outfit} in {scene} at the beginning.\n(a) {outfit1}\n(b) {outfit2}\n(c) {outfit3}",
        "Identify the order of outfits worn by the person shown up with {action} and wearing {outfit} in {scene} at the beginning.\n(a) {outfit1}\n(b) {outfit2}\n(c) {outfit3}",
    ],
    "later_to_start": [
        "What is the chronological order of the following outfits worn by the person who was seen at the end {action} and wearing {outfit} in {scene}?\n(a) {outfit1}\n(b) {outfit2}\n(c) {outfit3}",
        "Arrange the following outfits in the order worn by the person {action} and wearing {outfit} in {scene} at the end.\n(a) {outfit1}\n(b) {outfit2}\n(c) {outfit3}",
        "In what order did the person {action} and wearing {outfit} in {scene} at the end, worn through the video?\n(a) {outfit1}\n(b) {outfit2}\n(c) {outfit3}",
        "Put the following outfits in the order worn by the person {action} and wearing {outfit} in {scene} at the end.\n(a) {outfit1}\n(b) {outfit2}\n(c) {outfit3}",
        "Identify the order of outfits worn by the person shown up with {action} and wearing {outfit} in {scene} at the end.\n(a) {outfit1}\n(b) {outfit2}\n(c) {outfit3}",
    ]
}