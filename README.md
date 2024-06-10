# Residual Error Probability Simulator

![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [License](#license)
- [Authors](#authors)
- [Contacts](#contacts)

## Introduction

The residual error probability simulator evaluates the residual error probability of a given protocol (with the corresponding packet structure) with its given error detection mechanism. The simulator has a built-in CRC (Cyclic Redundancy Check) as an error detection method. It performs a Monte Carlo simulation injecting errors in the input frame data structure and evaluates whether such errors are detectable or not. The Monte Carlo simulation is enriched with Importance Sampling.

## Prerequisites

- [Python](https://www.python.org/downloads/) (version 3.6 or higher)
- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)

## Usage

In order to run the simulation, just type the following.
```sh
python3 main.py <tests/test_file.json> results/<output_directory>
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors ##

Paolo Guarino, Filippo Nevi, Paolo Dai Pra, Davide Quaglia, Tiziano Villa

*Department of Computer Science, University of Verona, Italy*

## Contacts

For any questions, suggestions, or feedback, feel free to contact:

- **Paolo Guarino** - [paolo.guarino_01@studenti.univr.it](mailto:paolo.guarino_01@studenti.univr.it)
- **Filippo Nevi** - [filippo.nevi@studenti.univr.it](mailto:filippo.nevi@studenti.univr.it)

Project Link: [https://github.com/guarinopaolo/residual-error-probability-simulator](https://github.com/guarinopaolo/residual-error-probability-simulator)
