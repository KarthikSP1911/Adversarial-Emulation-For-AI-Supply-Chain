"""Format-specific static scanners.

Each sub-module exposes a single ``scan(path: Path) -> list[Finding]``
function that inspects one file and returns zero or more :class:`Finding`
objects.  They are collected by :mod:`modelscan.static`.

Modules
-------
- :mod:`.torch`     – PyTorch ``.pt`` / ``.pth`` / ``.ckpt`` containers
- :mod:`.keras`     – Keras ``SavedModel`` / ``.keras`` / ``.h5`` bundles
- :mod:`.onnx`      – ONNX ``*.onnx`` graphs
- :mod:`.gguf`      – GGUF ``*.gguf`` LLM checkpoints
- :mod:`.numpy`     – NumPy ``*.npz`` / ``*.npy`` arrays
"""
