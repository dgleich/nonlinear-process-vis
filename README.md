Visualizing Nonlinear Processes on Graphs
=========================================

This repository describes (in code and rationale) a methodology that we explained
in

    Flow-based Algorithms for Improving Clusters: A Unifying Framework,
    Software, and Performance. Kimon Fountoulakis, Meng Liu,
    David F. Gleich, Michael W. Mahoney. https://arxiv.org/abs/2004.09608

The idea is that we can generalize various spectral or eigenvector-based
visualization techniques by exploiting a relationship between them and
random samples of the SVD. This is described above in more detail. Also, see
the video (link coming soon).

To understand this repository, please read

    - `live-notes.txt` *A set of overview notes we took as we proceeded*
    - `live-spectral-vis.jl` *This describes the spectral scenario*
    - `live-flow-intro.jl` *This describes the nonlinear process*
    - `live-flow-problem.jl` *This describes the issue we try and fix*
    - `live-sampled-vis.jl` **This describes our idea!**
    - `live-fashion-mnist.jl` *Another example of our idea*

There are two graphs used in this experiment. They are listed in the data
section.
