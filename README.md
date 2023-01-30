# mosap

[![License BSD-3](https://img.shields.io/pypi/l/mosap.svg?color=green)](https://github.com/minhtran1309/mosap/LICENSE)
[![GitHub Pages](https://img.shields.io/badge/docs-sphinx-blue)](https://github.com/BiomedicalMachineLearning/MOSAP)
[![PyPI](https://img.shields.io/pypi/v/mosap.svg?color=green)](https://pypi.org/project/mosap)
[![Python Version](https://img.shields.io/pypi/pyversions/mosap.svg?color=green)](https://python.org)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/mosap)](https://napari-hub.org/plugins/mosap)

# MOSAP(Multi-Omics Spatial Analysis Platform)

## Installation



The package is developed in Python and require napari to run the GUI

A popular way to install Python is via the [Anaconda](https://www.anaconda.com/products/individual) platform. 

After conda is successfully installed, you can create a new environment to run Python from terminal

```
conda create -y -n mosap -c conda-forge python=3.9
conda activate mosap
```
Your command line should display `(mosap)` label on it.

Prerequisite
you need to install `napari` to properly run mosap package.

```
pip install "napari[all]"
```



You can install `mosap` via [pip]:

    pip install mosap


## License

Distributed under the terms of the [BSD-3] license,
"mosap" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/

## Acknowledge

This [napari] plugin was generated using [@napari]'s [cookiecutter-napari-plugin] template.