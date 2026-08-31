# Related work

## How to read this file

Citations are marked by how far they have been checked:

* **`[KB nnn]`** — the paper is in the local library at `../3d-kb` under that id,
  and any claim attributed to it below was read out of the parsed text. Quotes
  are literal.
* **`[unverified]`** — written from memory. Title and first author are probably
  right; **venue and year should be checked before this goes anywhere public.**
  Nothing in this repo's code or claims depends on these.

The library has 603 papers but its `enrich` stage has not run, so it holds no
years or venues at all — which is why even the verified entries below carry no
year. Do not copy this file into a paper without a verification pass.

To move an entry from `[unverified]` to `[KB]`, drop the PDF into the library and
re-run its ingest.

---

## The gap this project addresses

The claim is narrow and worth stating precisely: **systems that ground
projective spatial language compute "left of" from object coordinates without
committing to, or reporting, a reference frame.** Two of the closest recent works
show this directly.

**SpatialRGPT** `[KB 603]` builds a 3D scene graph and defines its relation set
as:

> "Relative relations contain left, right, above, below, behind, front, wide,
> thin, tall, short, big, and small. […] We then traverse all the object nodes
> and use the point cloud centroids and bounding boxes to calculate their
> spatial relationships."
> — *SpatialRGPT: Grounded Spatial Reasoning in Vision Language Models*, p.5

So `left` and `right` are computed from centroids and boxes, with no reference
frame named anywhere in that description. The paper does treat one kind of
ambiguity, but as a *depth* problem rather than a frame problem: it reports gains
"in scenarios where relative depth information can be used to resolve
ambiguities, such as distinguishing between behind/front" (p.9). Depth resolves
which object is nearer the camera. It does not tell you whether the speaker
meant the camera's front or the sofa's.

**SpatialVLM** `[KB 602]` generates templated QA pairs including
"Given two objects A and B, which is more towards the left?" and
"Find out how far A is positioned behind B **relative to the camera**"
(p.4). The frame is the camera's, and it is stated inside one template string
rather than as a modelling commitment. Its benchmark is 331 qualitative and 215
quantitative human-annotated pairs (p.6) — labelled for the answer, not for
which frame the question assumes.

Neither is careless work; both are solving a different problem. The point is that
the frame is left implicit, so a wrong answer caused by a frame mismatch is
indistinguishable from one caused by perception — which is exactly what this
repo's failure attribution separates.

A supporting negative result: a search of all 603 papers for reference frames in
spatial language ("egocentric allocentric perspective left right ambiguity
viewer-centric object-centric") returns nothing on topic — the top hits are
image matching and egocentric *video*. That is evidence about this library, not
about the literature, and the linguistics work below exists and matters. But it
does suggest the 3D-vision reading list does not currently carry it.

---

## 1. Reference frames in spatial language

The three-way distinction this project is built on — intrinsic, relative,
absolute — is not new; it is standard in linguistics and cognitive science and
has been for decades. The contribution here is making it operational in a 3D
pipeline and measuring it, not discovering it.

* **Levinson, S. C. — *Space in Language and Cognition: Explorations in
  Cognitive Diversity*.** Cambridge University Press. `[unverified]`
  The canonical treatment of the intrinsic / relative / absolute trichotomy and
  of the cross-linguistic variation in which frame a language prefers. This
  repo's `intrinsic`, `egocentric` and `world` frames map onto that trichotomy;
  `addressee` is the mirrored relative reading Levinson also discusses.
* **Carlson-Radvansky, L. A. and Irwin, D. E. — "Frames of reference in vision
  and language: Where is above?"** *Cognition*. `[unverified]`
  Experimental evidence that speakers switch between frames for the *same*
  preposition depending on the reference object. Directly relevant to the
  asymmetry the policy in `sqe/frames/policy.py` encodes.
* **Talmy, L. — "How language structures space".** `[unverified]`
  Figure/ground asymmetry in spatial descriptions, which is why relations in
  this repo are directed (target, anchor) rather than symmetric.
* **Retz-Schmidt, G. — "Various Views on Spatial Prepositions".** *AI Magazine*.
  `[unverified]` An early AI treatment of deictic vs intrinsic readings.
* **Tversky, B. — "Spatial perspective in descriptions".** `[unverified]`
* **Herskovits, A. — *Language and Spatial Cognition*.** `[unverified]`
  On the vagueness and context-dependence of spatial prepositions, which is the
  reason ambiguity here is an output rather than an error.
* **Moratz, R. and Tenbrink, T. — spatial reference in human-robot
  interaction.** `[unverified]` The HRI literature on perspective-taking, where
  getting the frame wrong has consequences and is therefore taken seriously.
* **Trafton, J. G. et al. — perspective-taking with robots.** `[unverified]`

**What is different here.** These works establish that the frames exist and that
people switch between them. None of them, as far as I know, gives a 3D pipeline a
frame-selection policy, an availability test (can this anchor's front even be
estimated?), or a benchmark labelled by frame. That is the gap.

## 2. 3D referring-expression grounding

The task closest to this project's front end. **None of these are in the local
library**, which is worth fixing.

* **Chen, D. Z., Chang, A. X., Nießner, M. — "ScanRefer: 3D Object Localization
  in RGB-D Scans using Natural Language".** `[unverified]`
  The standard 3D grounding benchmark on ScanNet. Free-form descriptions, gold
  target box. No frame annotation.
* **Achlioptas, P. et al. — "ReferIt3D: Neural Listeners for Fine-Grained 3D
  Object Identification in Real-World Scenes"** (Nr3D and Sr3D). `[unverified]`
  Sr3D is machine-generated from spatial relation templates and so inherits
  whatever frame convention the generator used; Nr3D is human, and therefore
  contains a mixture of frames that is not labelled. This is precisely the
  situation this repo's benchmark is designed to un-mix.
* **Goyal, A., Yang, K., Yang, D., Deng, J. — "Rel3D: A Minimally Contrastive
  Benchmark for Grounding Spatial Relations in 3D".** `[unverified]`
  The nearest relative in spirit: minimally contrastive pairs isolating a single
  spatial relation. Rel3D isolates *whether* a relation holds; this project
  isolates *whose frame* it holds in.
* **Wang, T. et al. — "EmbodiedScan"**, a multi-modal ego-centric 3D perception
  and grounding benchmark. `[unverified]`
* **Zhang, Y. et al. — "Multi3DRefer"**, grounding with zero, one or many
  targets. `[unverified]` Relevant to this repo's handling of ambiguous queries
  with several acceptable answers.
* **Chen, S. et al. — language-conditioned spatial relation reasoning for 3D
  object grounding.** `[unverified]`
* **Yang, J. et al. — "LLM-Grounder"**, zero-shot 3D grounding with an LLM as
  the reasoning layer. `[unverified]` Comparable to this repo's LLM parser
  condition, though here the LLM only parses and never resolves geometry.
* **Zhu, Z. et al. — "3D-VisTA"**, pre-trained transformer for 3D vision and
  text alignment. `[unverified]`

**What is different here.** These evaluate end-to-end accuracy against a gold
box. Because a failure can come from the detector, the language model or the
frame, and only the first two are ever separated, the frame contribution is
invisible. This repo adds the gold *frame* label and the counterfactual
attribution that uses it.

## 3. Spatial reasoning in vision-language models

* **SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning
  Capabilities** `[KB 602]` — see the quotes above.
* **SpatialRGPT: Grounded Spatial Reasoning in Vision Language Models**
  `[KB 603]` — see the quotes above. Its region-plugin plus depth architecture
  is the strongest recent baseline for relative direction and distance.
* **SpatialLM: Training Large Language Models for Structured Indoor Modeling**
  `[KB 604]` — outputs structured layout (walls, doors, windows) plus objects
  from point clouds. Its wall and layout output is the kind of structure this
  repo's `sqe/geom/room.py` estimates in order to build the allocentric frame,
  and would be a better source for it than my Manhattan fit.

**What is different here.** These put the spatial reasoning inside a learned
model, so the frame convention is whatever the training data implied and cannot
be inspected, forced, or reported. Here the frame is an explicit object with a
confidence and a provenance, and the resolver can be *made* to answer under a
frame you choose — which is what the fixed-frame baselines in the report are.

## 4. Open-vocabulary 3D scene understanding

The perception component. Explicitly not the contribution, and replaceable.

* **OpenMask3D: Open-Vocabulary 3D Instance Segmentation** `[KB 583]`
  The design this repo's `openvocab3d` backend follows: class-agnostic 3D
  instance proposals, then per-instance CLIP features aggregated from the frames
  that best see each mask. The difference is that OpenMask3D uses a learned
  proposal network and this repo uses geometric over-segmentation, which is why
  its recall is modest and why that number is reported rather than assumed.
* **OpenScene: 3D Scene Understanding with Open Vocabularies** `[KB 579]`
  Per-point features co-embedded with text and images, enabling arbitrary text
  queries over a point cloud — the dense-feature alternative to per-instance
  embeddings. A per-point field is a poor fit for the relations here, which need
  discrete objects with extents and orientations.
* **LERF: Language Embedded Radiance Fields** `[KB 580]` and
  **LangSplat: 3D Language Gaussian Splatting** `[KB 491]`
  Language fields in NeRF and in Gaussian splats. Same objection for this task:
  a relevancy field has no object boundary, so "the second mug from the left"
  has nothing to count.
* **Segment Anything in 3D with NeRFs** `[KB 585]`
* **SAM 3D: 3Dfy Anything in Images** `[KB 377]`
* **Mask3D**, learned 3D instance segmentation `[unverified]` — the obvious
  replacement for the geometric proposal stage; it plugs in at
  `scene_from_masks`.
* **ConceptGraphs**, open-vocabulary 3D scene graphs for robotic perception
  `[unverified]` — the closest work on the *representation* side: it also builds
  an object-level graph with open-vocab labels and inter-object relations. Worth
  reading carefully for how it phrases relations, since if it also computes
  left/right from centroids it is another instance of the gap.
* **ConceptFusion**, open-set multimodal 3D mapping `[unverified]`
* **OVIR-3D**, open-vocabulary 3D instance retrieval `[unverified]`

## 5. 3D scene graphs and spatial perception

* **Hydra: A Real-time Spatial Perception System for 3D Scene Graph
  Construction and Optimization** `[KB 435]`
* **3D Dynamic Scene Graphs: Actionable Spatial Perception with Places, Objects,
  and Humans** `[KB 434]`
* **Kimera: an Open-Source Library for Real-Time Metric-Semantic Localization
  and Mapping** `[KB 153]`, **Kimera-Multi** `[KB 154]`

**What is different here.** These build the scene graph — the layer this project
consumes — and their edges are metric and topological (contains, is-near,
supports). They do not attempt projective relations, and so do not run into the
frame problem. A natural integration: use Hydra's graph as the scene and this
repo's resolver as its query layer.

## 6. Datasets

* **ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes** `[KB 382]`
* **ScanNet++: A High-Fidelity Dataset of 3D Indoor Scenes** `[KB 551]`
  The primary target here. Note the two format facts established by probing
  rather than by reading: `iphone/depth.bin` is per-frame LZ4, and
  `aligned_pose` is OpenCV camera-to-world in the mesh frame. See
  `sqe/data/scannetpp.py`.
* **ARKitScenes: A Diverse Real-World Dataset For 3D Indoor Scene
  Understanding** `[KB 552]`
  Supported as a secondary dataset specifically because its 3DOD annotations are
  *oriented* boxes, so the intrinsic frame can be built from annotated
  orientation rather than an estimate. An annotated box fixes the orientation
  axis but not which face is the front, so the sign is still estimated.
* **Matterport3D** `[KB 548]`, **Habitat-Matterport 3D (HM3D)** `[KB 558]`
  Larger-scale indoor environments; candidates for extending the benchmark
  beyond five scenes.

## 7. Components used directly

* **Radford, A. et al. — "Learning Transferable Visual Models From Natural
  Language Supervision"** (CLIP). `[unverified]`
  Used for open-vocabulary instance labelling via multi-view crops
  (`sqe/perception/clip_features.py`), with a prompt ensemble.
* **Kirillov, A. et al. — "Segment Anything"** (SAM). `[unverified]`
  Not currently used; the 2D-mask-lifting backend it would serve is the
  alternative to the geometric proposal stage.
* **Felzenszwalb, P. F. and Huttenlocher, D. P. — "Efficient Graph-Based Image
  Segmentation".** `[unverified]`
  The over-segmentation in `sqe/perception/proposals.py` is this algorithm run
  on the mesh edge graph with normal and colour edge weights — the same
  construction ScanNet's `segmentator` uses for its over-segmentation.
* **Toussaint, G. — rotating calipers** for the minimum-area enclosing
  rectangle. `[unverified]`
  `sqe/geom/obb.py` fits the yaw-only oriented box this way, which is what makes
  the fit agree with ScanNet++'s own annotation boxes to sub-micrometre median
  centre error.

---

## Positioning, in one paragraph

Open-vocabulary 3D segmentation is a crowded, well-served area, and this project
treats it as a replaceable component and measures its quality rather than
claiming it. 3D referring-expression grounding has good benchmarks that evaluate
end-to-end accuracy against a gold box. The spatial-language literature has known
for decades that projective terms are frame-relative and that speakers switch
frames systematically. What is missing is the join: a resolver that treats the
reference frame as an explicit, inspectable, forcible parameter; a benchmark
labelled by *which frame the sentence means* and by *whether it is ambiguous at
all*; and an error attribution that can therefore say how much of the remaining
failure is a frame-convention mistake rather than a perception mistake.

## Papers to add to the library

In rough order of usefulness for writing this up. None of the code depends on
them; they matter for positioning and for checking that the gap claim survives
contact with the closest work.

1. ScanRefer, and ReferIt3D (Nr3D / Sr3D) — the benchmarks to compare against
2. Rel3D — nearest in spirit on the relation-isolation idea
3. ConceptGraphs — check how it phrases inter-object relations
4. Levinson, *Space in Language and Cognition* — the frame trichotomy
5. Carlson-Radvansky and Irwin, "Where is above?" — the frame-switching evidence
6. Mask3D — the drop-in proposal network
7. EmbodiedScan, Multi3DRefer — recent grounding benchmarks
8. CLIP and Felzenszwalb — components, for correct attribution
