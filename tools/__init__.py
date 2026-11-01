"""
tools -- standalone utilities. Not part of the pipeline, and never invoked by it.

    python -m tools.download_data       fetch the datasets into data/download/

Everything here is run by hand, does one job, and leaves its result on disk for the pipeline
to find. The separation matters most for downloading: no trainer reaches the network, so a
training run either finds its corpus locally or stops, and can never quietly fetch several
gigabytes and train on something nobody chose.
"""
