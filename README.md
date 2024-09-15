# Auto-Landmark
This is open source toolkit for speech landmark detection
Here's the English version of the introduction:

---

## Introduction

This toolkit provides an efficient and accurate method for extracting acoustic landmarks, assisting researchers and developers in identifying key acoustic events in speech signal processing. The definition of landmarks is based on the paper ["Landmark detection for distinctive feature-based speech recognition"](https://pubs.aip.org/asa/jasa/article/100/5/3417/559799/Landmark-detection-for-distinctive-feature-based), offering a framework for landmark detection grounded in distinctive speech features. Our toolkit builds upon the methodology proposed in the paper ["Auto-Landmark: Acoustic Landmark Dataset and Open-Source Toolkit for Landmark Extraction"](https://arxiv.org/abs/2409.07969), implementing an open-source solution that allows users to extract and analyze acoustic landmarks in their own projects seamlessly.


Here’s an additional section to include before the "Basic Use" section, based on the information in the table:

---


## Basic Use

To extract acoustic landmarks from an audio file, you can use the following example. First, import the `extract_all_landmarks` function and pass the path of the audio file as an argument:

```python
from methods.Basic.Landmarks_func import extract_all_landmarks

# Example of extracting landmarks from an audio file
landmarks = extract_all_landmarks("path/to/your/audiofile.wav")
```

In this example, replace `"path/to/your/audiofile.wav"` with the actual file path of the audio file you want to process. The function will return the detected landmarks, which can then be used for further analysis.

Here's a general description of the extracted landmarks:

---

The toolkit extracts various acoustic landmarks from an audio file, providing their occurrence times within the signal. Each landmark type is represented by a key, with an associated list of timestamps where these landmarks were detected. The keys correspond to specific types of acoustic events, such as:

- **'g+':** Glottal Onsets
- **'g-':** Glottal Offsets
- **'s+':** Syllabic Onsets
- **'s-':** Syllabic Offsets
- **'b+':** Burst Onsets
- **'b-':** Burst Offsets
- **'v+':** Voiced Frication Onsets
- **'v-':** Voiced Frication Offsets
- **'f+':** Frication Onsets
- **'f-':** Frication Offsets
- **'p+':** Periodicity Onsets
- **'p-':** Periodicity Offsets

Each list contains the times (in seconds) at which the corresponding landmarks were detected in the audio file. If no landmarks of a specific type are found, an empty list is returned.

Here's the "Recipe Use" section indicating that it is currently under development:

---

## Recipe Use

We are actively developing detailed recipes for various use cases, including extracting landmarks from different datasets, customizing detection parameters, and integrating with other speech processing pipelines. These recipes will provide step-by-step instructions to help users tailor the toolkit for their specific needs. Stay tuned for updates and new features in upcoming releases!